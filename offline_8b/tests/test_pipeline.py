from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from offline_8b.common import MODEL_ID, canonical, compatible_manifest, digest, file_hash, read_db, write_json
from offline_8b.encode import (Encoder, check_shards, encode_pending, init_output, length_report,
                               resolve_model, tokenize_record, validate_vectors)
from offline_8b.loader import AccountEmbeddings
from offline_8b.prepare import load_split_labels, run
from offline_8b.text import Cleaner
from offline_8b.verify import verify


class CharacterTokenizer:
    """Test tokenizer: deterministic tokens make head/tail boundaries inspectable."""
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


class FakeEncoder:
    """Only for storage/integrity tests; never exposed by production CLI."""
    def __init__(self):
        self.calls = 0
    def encode(self, rows):
        self.calls += len(rows)
        matrix = np.zeros((len(rows), 4096), dtype=np.float32)
        for i, row in enumerate(rows):
            matrix[i, sum(row) % 4096] = 1
        return matrix


class TextTests(unittest.TestCase):
    def setUp(self):
        self.cleaner = Cleaner()

    def test_multilingual_identity_and_url(self):
        raw = 'ＡＢＣ &amp; 中文 العربية 👩\u200d💻 #Spam @Alice a@b.com https://sub.example.co.uk/invite/SECRET?token=PRIVATE#frag\x00'
        item = self.cleaner.clean(raw)
        self.assertIn('ABC & 中文 العربية 👩\u200d💻 #Spam <USER> <EMAIL> <URL>', item['text'])
        self.assertEqual(item['url_count'], 1)
        self.assertEqual(item['urls'][0]['main_domain'], 'example.co.uk')
        self.assertEqual(item['urls'][0]['normalized_url'], 'https://sub.example.co.uk/')
        self.assertNotIn('SECRET', item['text'])
        self.assertNotIn('PRIVATE', item['urls'][0]['normalized_url'])

    def test_html_and_raw_url(self):
        item = self.cleaner.clean('<p>Hello</p><script>secret</script><a href="https://a.org/?a=1&amp;b=2">你好</a>')
        self.assertNotIn('secret', item['text'])
        self.assertIn('你好', item['text'])
        self.assertEqual(item['url_count'], 1)
        self.assertEqual(item['urls'][0]['raw_url'], 'https://a.org/?a=1&amp;b=2')

    def test_control_empty_and_no_lowercase(self):
        self.assertEqual(self.cleaner.clean('  <b> </b>\x00\u200b')['valid_mask'], 0)
        self.assertEqual(self.cleaner.clean('UPPER\n\n\n中文')['text'], 'UPPER\n\n中文')

    def test_schemeless_query_and_non_http_url(self):
        item = self.cleaner.clean('example.org?token=SECRET#frag ftp://files.example.com/id @alice.')
        self.assertEqual(item['text'], '<URL> <URL> <USER>.')
        self.assertEqual(item['url_count'], 2)
        self.assertNotIn('SECRET', item['text'])

    def test_head_tail_tokens_and_instruction(self):
        tokenizer = CharacterTokenizer()
        ids, meta = tokenize_record(tokenizer, 'abcdefghij', 6, False)
        self.assertEqual(ids, list(map(ord, 'abchij')))
        self.assertEqual(meta, dict(original_token_count=10, retained_token_count=6, is_truncated=True))
        ids, meta = tokenize_record(tokenizer, 'X' * 1000 + '尾', 300, True)
        self.assertTrue(''.join(map(chr, ids)).startswith('Instruct: '))
        self.assertEqual(ids[-1], ord('尾'))
        self.assertEqual(len(ids), 300)
        self.assertEqual(tokenize_record(tokenizer, '', 512)[0], [])

    def test_oom_reduces_only_batch_size(self):
        from types import SimpleNamespace
        class OOM(RuntimeError):
            pass
        encoder = object.__new__(Encoder)
        encoder.dimension, encoder.batch_size = 4096, 8
        encoder.torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None))
        def batch(rows):
            if len(rows) > 2:
                raise OOM('simulated CUDA OOM')
            return FakeEncoder().encode(rows)
        encoder.batch = batch
        rows = [[x] * (x + 1) for x in range(8)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = encoder.encode(rows)
        np.testing.assert_array_equal(result, FakeEncoder().encode(rows))
        self.assertEqual(encoder.batch_size,2)
        self.assertEqual(encoder.dimension,4096)

    def test_model_cache_root_and_mrl_config_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'hub'/'snapshots'/'abc123'
            (model/'1_Pooling').mkdir(parents=True)
            write_json(model/'config.json',{'model_type':'qwen3','hidden_size':4096,'num_hidden_layers':36,'vocab_size':151665})
            write_json(model/'modules.json',[{'type':'sentence_transformers.models.Transformer'},
                                             {'type':'sentence_transformers.models.Pooling'}])
            write_json(model/'1_Pooling/config.json',{'pooling_mode_lasttoken':True,'pooling_mode_mean_tokens':False,
                                                     'word_embedding_dimension':4096})
            path, revision, _ = resolve_model(root/'hub')
            self.assertEqual(path,model)
            self.assertEqual(revision,'abc123')
            cache = root/'output'
            cache.mkdir()
            compatible_manifest(cache,'manifest.json',{'dimension':4096},False)
            with self.assertRaises(ValueError):
                compatible_manifest(cache,'manifest.json',{'dimension':2560},True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.db'
        db = sqlite3.connect(self.source)
        db.executescript('''
        CREATE TABLE agents(id TEXT PRIMARY KEY, name TEXT, description TEXT, x_handle TEXT, x_bio TEXT, last_updated TEXT);
        CREATE TABLE posts(id TEXT PRIMARY KEY,title TEXT,content TEXT,url TEXT,author_id TEXT,created_at TEXT);
        CREATE TABLE comments(id TEXT PRIMARY KEY,content TEXT,author_id TEXT,created_at TEXT);
        ''')
        db.executemany('INSERT INTO agents VALUES(?,?,?,?,?,?)', [
            ('a','Alice','中文 👩‍💻','SameOwner','Public bio','2026-01-01'),
            ('b','Bob',None,'@SameOwner',None,'2026-01-02'),
            ('c','Carol','<p> </p>',None,'','2026-01-03')])
        db.executemany('INSERT INTO posts VALUES(?,?,?,?,?,?)', [
            ('p1','标题','Buy https://example.org/i/TOKEN?q=x',None,'a','2026-01-01'),
            ('p2','标题','Buy https://example.org/i/TOKEN?q=x',None,'a','2026-01-02'),
            ('p3','Long','多语言' * 200,None,'b','2026-01-03'),
            ('p4','','',None,'c','2026-01-04'),
            ('p5','unassigned','kept',None,'d','2026-01-05'),
            ('p6','orphan','kept',None,None,'2026-01-06')])
        db.executemany('INSERT INTO comments VALUES(?,?,?,?)', [
            ('c1','@somebody A@B.org #保留', 'a','2026-01-01'),
            ('c2','', 'b','2026-01-02')])
        db.commit()
        db.close()
        self.splits = self.root / 'splits.json'
        self.labels = self.root / 'accounts.json'
        write_json(self.splits, {'train':['a'], 'validation':['b'], 'test':['c']})
        write_json(self.labels, [{'account_id':a,'label':v} for a,v in [('a',1),('b',0),('c',1)]])
        self.prepared = self.root / 'prepared'
        self.output = self.root / 'output'
        self.args = argparse.Namespace(source_db=str(self.source),output_dir=str(self.prepared),
            splits=str(self.splits),accounts_json=str(self.labels),label_db=None,page_size=2,resume=False)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.args)

    def setup_encoding(self):
        self.output.mkdir()
        (self.output / 'shards').mkdir()
        db = sqlite3.connect(self.output / 'index.sqlite', uri=True)
        init_output(db)
        with read_db(self.prepared / 'index.sqlite') as source:
            for rid, kind, payload in source.execute('SELECT record_id,source_type,payload FROM records ORDER BY record_id'):
                item = json.loads(payload)
                ids, info = tokenize_record(CharacterTokenizer(), item['text'], 64, False)
                key = digest(canonical({'text_hash':item['text_hash'], 'ids':ids}))
                db.execute('INSERT INTO tokens VALUES(?,?,?,?,?,?)', (rid,key,zlib.compress(canonical(ids).encode()),
                    info['original_token_count'],info['retained_token_count'],kind))
        db.commit()
        return db

    def finish_manifest(self, db):
        import shutil
        for name in ('accounts.jsonl','splits.json','labels.json','owner_split_audit.json'):
            shutil.copyfile(self.prepared/name,self.output/name)
        write_json(self.output/'embedding_manifest.json', {'complete':True,'records':18,'accounts':5,
            'config':{'model_name':MODEL_ID,'embedding_dimension':4096,'max_length':64,'profile_max_length':64,
                      'prepared_index_sha256':file_hash(self.prepared/'index.sqlite')}})

    def test_end_to_end_all_instances_masks_and_resume(self):
        before = file_hash(self.source)
        self.prepare()
        self.assertEqual(before, file_hash(self.source))
        self.args.resume = True
        self.prepare()
        db = self.setup_encoding()
        encoder = FakeEncoder()
        with contextlib.redirect_stdout(io.StringIO()):
            encode_pending(self.prepared,self.output,db,encoder,4096,2)
        calls = encoder.calls
        encode_pending(self.prepared,self.output,db,encoder,4096,2)
        self.assertEqual(encoder.calls,calls)
        self.assertLess(calls, db.execute('SELECT COUNT(*) FROM instances WHERE valid_mask=1').fetchone()[0])
        check_shards(self.output,db)
        self.finish_manifest(db)
        db.close()
        with contextlib.redirect_stdout(io.StringIO()):
            report = verify(self.output,self.prepared,1)
        self.assertTrue(report['passed'])
        loader = AccountEmbeddings(self.output)
        self.assertEqual(loader['a']['post_embeddings'].shape,(2,4096))
        np.testing.assert_array_equal(loader['a']['post_embeddings'][0],loader['a']['post_embeddings'][1])
        self.assertEqual(loader['b']['twitter_profile_mask'],0)
        self.assertEqual(loader['c']['moltbook_profile_mask'],0)
        self.assertEqual(loader['c']['comment_embeddings'].shape,(0,4096))
        self.assertEqual(loader['d']['split'],'unassigned')
        self.assertEqual(loader['a']['post_metadata'][0]['exact_duplicate_count'],1)
        self.assertEqual(loader['a']['twitter_owner_group'],loader['b']['twitter_owner_group'])
        loader.close()
        with self.assertRaises(RuntimeError):
            socket.create_connection(('example.com',443))

    def test_interrupted_shard_retry(self):
        self.prepare()
        db = self.setup_encoding()
        encoder = FakeEncoder()
        import offline_8b.encode as module
        real_replace = module.os.replace
        calls = [0]
        def interrupted(src,dst):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('simulated interruption after vector rename')
            return real_replace(src,dst)
        with patch.object(module.os,'replace',side_effect=interrupted):
            with self.assertRaises(OSError):
                encode_pending(self.prepared,self.output,db,encoder,4096,2)
        db.close()
        db = sqlite3.connect(self.output/'index.sqlite',uri=True)
        with contextlib.redirect_stdout(io.StringIO()):
            encode_pending(self.prepared,self.output,db,encoder,4096,2)
        self.assertEqual(db.execute('SELECT COUNT(*) FROM instances').fetchone()[0],18)
        check_shards(self.output,db)
        db.close()

    def test_preprocessing_interruption_and_resume(self):
        import offline_8b.prepare as module
        real_insert = module.insert_record
        calls = [0]
        def interrupted(db, item):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError('simulated preprocessing interruption')
            return real_insert(db, item)
        with patch.object(module, 'insert_record', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                self.prepare()
        import gc
        gc.collect()
        self.args.resume = True
        self.prepare()
        with read_db(self.prepared/'index.sqlite') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM records').fetchone()[0],18)
            self.assertEqual(db.execute('SELECT COUNT(DISTINCT content_id) FROM records WHERE source_type="post"').fetchone()[0],6)
    def test_reject_overlap_and_changed_labels(self):
        self.prepare()
        self.args.resume = True
        write_json(self.labels,[{'account_id':a,'label':0} for a in 'abc'])
        with self.assertRaises(ValueError):
            self.prepare()
        write_json(self.splits,{'train':['a'],'validation':['a'],'test':['c']})
        with self.assertRaises(ValueError):
            load_split_labels(self.splits,self.labels)

    def test_corrupt_shard_is_rejected(self):
        self.prepare()
        db = self.setup_encoding()
        with contextlib.redirect_stdout(io.StringIO()):
            encode_pending(self.prepared,self.output,db,FakeEncoder(),4096,4)
        shard = self.output/'shards/shard-000000.npy'
        with shard.open('ab') as f:
            f.write(b'corruption')
        with self.assertRaises(ValueError):
            check_shards(self.output,db)
        db.close()

    def test_vector_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_vectors(np.full((1,4096),np.nan),[1],4096)
        with self.assertRaises(ValueError):
            validate_vectors(np.zeros((1,2560)),[0],4096)
        with self.assertRaises(ValueError):
            validate_vectors(np.ones((1,4096)),[1],4096)


if __name__ == '__main__':
    unittest.main()

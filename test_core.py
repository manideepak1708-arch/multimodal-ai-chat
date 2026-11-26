import pytest
from core import chunks, evidence_prompt, retrieve, answer_pdf

def test_chunks():
    assert chunks('  one   two  ',20,3)==['one two']
    assert chunks('abcdefghi',4,1)==['abcd','defg','ghi']
    assert chunks('')==[]
    with pytest.raises(ValueError): chunks('x',4,4)

def test_retrieval():
    class Model:
        def embed(self, texts): return [[1.0,0.0]]
    class Collection:
        def count(self): return 1
        def query(self,**kwargs):
            assert kwargs['n_results']==1
            return {'documents':[['evidence']],'metadatas':[[{'page':2}]]}
    assert retrieve(Collection(),'query',Model())==[{'page':2,'text':'evidence'}]

def test_prompt():
    assert '[Page 3] example' in evidence_prompt('question',[{'page':3,'text':'example'}])
    assert answer_pdf(None,'question',[])=='No relevant excerpts were retrieved.'

from src.data import Vocabulary


def test_tokenize_and_encode():
    v = Vocabulary(min_freq=1)
    v.add_sentence('A cat sits on the mat.')
    v.build()
    idxs = v.encode('A cat sits on the mat.')
    assert isinstance(idxs, list)
    assert len(idxs) > 0

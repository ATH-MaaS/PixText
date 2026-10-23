import numpy as np
import pytest

from comfyui.ocr import crop_quad, decode_ctc, model_files


def test_ctc_repeats_blanks_and_padding():
    vocabulary = ["blank", "o", "b", "k", "X"]
    # b oo k with repeat frames, and an X entirely in right-hand batch padding.
    ids = [2, 2, 1, 1, 0, 1, 3, 3, 4, 4]
    probabilities = np.eye(5, dtype=np.float32)[ids]
    text, score = decode_ctc(probabilities, vocabulary, 8)
    assert text == "book"
    assert score == 1


def test_ctc_dictionary_mismatch_is_explicit():
    with pytest.raises(ValueError, match="dictionary"):
        decode_ctc(np.eye(4)[[3]], ["blank", "a"], 1)


def test_empty_ctc_returns_no_text():
    assert decode_ctc(np.eye(2)[[0, 0]], ["blank", "a"], 2) == ("", 0)


def test_perspective_crop_samples_the_selected_word():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    image[10:31, 20:61] = [10, 120, 240]
    crop = crop_quad(image, np.float32([[20, 10], [60, 10], [60, 30], [20, 30]]))
    assert crop.shape == (20, 40, 3)
    np.testing.assert_array_equal(crop[10, 20], [10, 120, 240])


def test_missing_models_do_not_trigger_downloads(tmp_path):
    with pytest.raises(ValueError, match="Install the optional OCR models"):
        model_files(tmp_path)

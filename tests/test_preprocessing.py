import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from processing.clean_text import clean_text, sliding_window_words 

def test_pipeline():
    print("Executing Preprocessing Verification Suite...\n")
    
    # Mock messy string representing Project Gutenberg noise
    mock_raw_text = """
    *** START OF THE PROJECT GUTENBERG EBOOK MOCK ***
    [Translator Note: This is a footnote that should disappear.]
    Footnotes
    Translator: Sir Edwin Arnold
    By the editor and translator
    Chapter I.
    1:1 In the beginning, Dhananjaya rode his chariot into the fray.
    1:2 Krishna stood beside him as they faced the opposing ranks.
    *** END OF THE PROJECT GUTENBERG EBOOK MOCK ***
    """
    
    # 1. Test Data Normalization
    cleaned = clean_text(mock_raw_text)
    print("--- 1. Regex Cleaning Test ---")
    print(f"Cleaned Text Output:\n\"{cleaned}\"\n")
    
    assert "START OF THE PROJECT" not in cleaned, "Failed to strip Gutenberg Header"
    assert "END OF THE PROJECT" not in cleaned, "Failed to strip Gutenberg Footer"
    assert "Translator Note" not in cleaned, "Failed to remove bracketed footnotes"
    assert "Footnotes" not in cleaned, "Failed to remove standalone footnote labels"
    assert "Sir Edwin Arnold" not in cleaned, "Failed to remove translator/editor header labels"
    assert "1:1" not in cleaned and "1:2" not in cleaned, "Failed to strip verse markers"
    assert "Chapter I." not in cleaned, "Failed to strip standalone chapter headings"
    print("✅ All Regex Normalization rules passed.\n")
    
    # 2. Test Sliding Window Logic
    print("--- 2. Sliding Window Structural Test ---")
    sample_clean_prose = "one two three four five six seven eight nine ten"
    
    # Window size 4, Stride 2
    # Expected Window 1: ['one', 'two', 'three', 'four']
    # Expected Window 2: ['three', 'four', 'five', 'six']
    # Expected Window 3: ['five', 'six', 'seven', 'eight']
    # Expected Window 4: ['seven', 'eight', 'nine', 'ten']
    
    windows = list(sliding_window_words(sample_clean_prose, window_size=4, stride=2))
    
    print(f"Input text length: {len(sample_clean_prose.split())} words")
    print(f"Generated {len(windows)} windows using Size=4, Stride=2:")
    for idx, win in enumerate(windows):
        print(f"  Window {idx + 1}: {win}")
        
    assert windows[0] == ['one', 'two', 'three', 'four'], "Window slicing alignment error"
    assert windows[1] == ['three', 'four', 'five', 'six'], "Stride calculation step error"
    assert windows[-1][-1] == 'ten', "Failed to reach the final token boundary"
    print("\n✅ Sliding window step and stride logic passed.")
    print("\nEverything works.")

if __name__ == "__main__":
    test_pipeline()
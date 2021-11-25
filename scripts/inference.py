import tensorflow as tf
from kmeseg.core.tokenize import Tokenizer, Segmentation


def inference(segmentation: Segmentation, text):
    text_pre = segmentation.preprocessing_text(text)
    pred_text, pred_text_join = segmentation.word_segmentation(text)
    
    print(text)
    print(text_pre)
    print(pred_text)
    print(pred_text_join)

def main():
    segmentor = Segmentation()
    tokenizer = Tokenizer()
    
    text = input('Enter IUPAC name to be segmentation: ')
    inference(segmentor, text)

if __name__ == '__main__':
    main()
from tibetan_wer.wer import wer

prediction = 'འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔'
reference = 'གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔'

wer_score = wer(prediction, reference)

print(f'WER Score: {wer_score}')
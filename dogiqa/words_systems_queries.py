
assess_standard_quality_sentence = '''
5: Excellent! The overall quality of the image is extremely excellent, and all aspects are perfect.
4: Good! The overall quality of the image is good, and it is satisfactory in many aspects.
3: Fair! The overall quality of the image is fair. There are certain merits but also some deficiencies.
2: Poor! The overall quality of the image is poor, with obvious defects.
1: Bad! The overall quality of the image is bad and hard to accept.
'''

assess_standard_quality_sentence_7 = '''
7: Perfect! The overall quality of the image is unparalleled, with every detail meeting the highest standards.
6: Excellent! The overall quality of the image is excellent, with all aspects being exemplary and without flaw.
5: Good! The overall quality of the image is good, and it is satisfactory in many aspects.
4: Fair! The overall quality of the image is fair. There are certain merits but also some deficiencies.
3: Bad! The overall quality of the image is bad, with noticeable shortcomings.
2: Poor! The overall quality of the image is poor, with obvious defects.
1: Very Bad! The overall quality of the image is very bad and hard to accept.
'''

assess_standard_quality_word_3 = '''
3: Excellent!
2: Fair!
1: Poor!
'''

assess_standard_quality_word_5 = '''
5: Excellent!
4: Good!
3: Fair!
2: Bad!
1: Poor!
'''

assess_standard_quality_word_7 = '''
7: Perfect!
6: Excellent!
5: Good!
4: Fair!
3: Bad!
2: Poor!
1: Very Bad!
'''

assess_standard_quality_word_9_2 = '''
Perfect!
Excellent!
Very Good!
Good!
Fair!
Bad!
Poor!
Awful!
Very Bad!
'''

assess_standard_quality_word_9 = '''
9: Perfect!
8: Excellent!
7: Very Good!
6: Good!
5: Fair!
4: Bad!
3: Poor!
2: Awful!
1: Very Bad!
'''

assess_standards_word={
    'quality_9':assess_standard_quality_word_9,
    'quality_7':assess_standard_quality_word_7,
    'quality_5':assess_standard_quality_word_5,
    'quality_3':assess_standard_quality_word_3,
}


def get_system_and_query(standard:str, n_word=-1):
    # assess_dim should be in ['quality','brightness','contrast','noise','sharpness','clearness']
    # assert assess_word in assess_words
    if standard == 'number':
        system = f'''You are a helpful assistant to help me evaluate the quality of the image. 
                    Please strictly follow the USER's format, otherwise the result will be invalid.'''
        query = f'please evaluate the quality of the image and score in [1,2,3,4,5,6,7]. Only tell me the number. Do not analysis the image.'
    
    if standard == 'word':
        if n_word != -1:
            assess_standard = assess_standards_word[f"quality_{n_word}"]
            eval_range=str(list(range(1, n_word+1)))
        else:
            assess_standard = assess_standards_word['quality_7']
            eval_range=str(list(range(1, 7+1)))
            
        system = f'''You are a helpful assistant to help me evaluate the quality of the image. You will be given standards about each quality level. The quality standard is list as follows:{assess_standard}
                    The higher the image quality, the higher the score should be.
                    Please strictly follow the USER's format, otherwise the result will be invalid.'''
        query = f'Please evaluate the quality of the image and score in quality. Only tell me the number. Do not analysis the image.'
      
    if standard == 'sentence':
        assess_standard = assess_standard_quality_sentence        
        system = f'''You are a helpful assistant to help me evaluate the quality of the image. You will be given standards about each {assess_word} level. The {assess_word} standard is list as follows:
                    {assess_standard}
                    Please strictly follow the USER's format, otherwise the result will be invalid.'''
        query = f'please evaluate the quality of the image and score in [1,2,3,4,5]. Only tell me the number. Do not analysis the image.'
    
    if standard == 'number_decimal':
        system = f'''You are a helpful assistant to help me evaluate the quality of the image. 
                    Your response must be a single number between 0 and 100, inclusive, with one decimal place. 
                    Do not give any explanation or description of the image. 
                    Simply provide the numeric score as your response. Any additional text will be considered invalid.'''
        query = f'Please evaluate the quality of the image with a score from 0 to 100, with one decimal place. Use a number between 0.0 and 100.0 to indicate your evaluation. For example, if you consider the image to be of very high quality, you might respond with "95.3". Conversely, if the image is of very poor quality, you might respond with "5.8". Your response should only be the score, without any other text.'
    
    if standard =='word_only':
        system = f'''You are a helpful assistant designed to evaluate the quality of an image. 
            Your response must be one of the following descriptive words that best represents the quality of the image:
            {assess_standard_quality_word_9_2}
            Do not give any explanation or description of the image. Simply provide the word that corresponds to your evaluation. Any additional text will be considered invalid.'''
        query = f'Please evaluate the quality of the image and respond with the most appropriate word from the list above. Your response should only be one word, without any other text.'
    
    return system, query



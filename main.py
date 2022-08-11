import tensorflow as tf

text = open('pianoabc.txt', 'r').read()
#악보를 문자 데이터로 바꾸는 abc notation

unique_text = list(set(text))
unique_text.sort()

#utilities
text_to_num = {}
num_to_text = {}

for value, key in enumerate(unique_text):
    text_to_num[key] = value
    num_to_text[value] = key

#원본 데이터 -> 숫자 데이터

num_text = []

for i in text:
    num_text.append(text_to_num[i])

#데이터 셋
X = []
Y = []

for i in range(0, len(num_text) - 25):
    X.append(num_text[i : i + 25])
    Y.append(num_text[i + 25])

import numpy as np

#print( np.array(X).shape )

#원 핫 인코딩

X = tf.one_hot(X, 31) #유니크 문자 개수
Y = tf.one_hot(Y, 31)

#model = tf.keras.models.Sequential([
#    tf.keras.layers.LSTM( 100, input_shape=(25, 31)),
#    tf.keras.layers.Dense(31, activation='softmax') #31개중 하나 예측
#])
#activation='softmax'와 loss = 'categorical_crossentropy'는 세트
#model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) #원핫 인코딩이 되어있으면 sparse 안써도됨

#model.fit(X, Y, batch_size=64, epochs=40, verbose=2) #64개 데이터 학습후에 w값 업데이트
#model.save('model1')

pModel = tf.keras.models.load_model('model1')

first_input = num_text[117 : 117 + 25]
first_input = tf.one_hot(first_input, 31)
first_input = tf.expand_dims(first_input, axis=0)

# guess_val = pModel.predict(first_input)
# guess_val = np.argmax(guess_val[0])

# 0. 첫입력값 만들기
# 1. Predict로 다음문자 예측
# 2. 예측한 다음문자 [] 저장하기
# 3. 첫입력값 앞에 짜르기
# 4. 예측한 다음문자 뒤에 넣기
# 5. 원핫 인코딩 하기, expand dims

music = []

for i in range(200):
    guess_val = pModel.predict(first_input)
    guess_val = np.argmax(guess_val[0])

    music.append(guess_val)
    next_input = first_input.numpy()[0][1:]

    one_hot_num = tf.one_hot(guess_val, 31)
    first_input = np.vstack([ next_input, one_hot_num.numpy() ])
    first_input = tf.expand_dims(first_input, axis=0)

music_text = []

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))
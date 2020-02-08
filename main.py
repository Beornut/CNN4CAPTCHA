import sqlite3, os, shutil, time
from PIL import Image
import numpy as np
import tensorflow as tf

imgs = os.listdir('std')[1000:3000] + os.listdir('std')[5000:6900]
test_imgs = os.listdir('std')[6900:]
imgs_len = len(imgs)
chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
chars_len = len(chars)
label_len = 4
img_w = 160
img_h = 60


def copy_and_label():
    path = r'1'
    conn = sqlite3.connect(os.path.join(path, 'label.db'))
    cursor = conn.cursor()
    [shutil.copyfile(os.path.join(path, i), os.path.join('all', '3_' + str(int(time.time())) + '_' +
                                                         cursor.execute('select label from label where name=?',
                                                                        (i,)).fetchone()[0] + '.png')) for i in
     os.listdir(path) if i[-1] == 'g']
    cursor.close()
    conn.close()


def reshape_and_gray():
    path = r'all'
    [Image.open(os.path.join(path, i)).convert('L').resize((img_w, img_h)).save(os.path.join('std', i)) for i in
     os.listdir(path)]


def get_batch(idx, size):
    X = np.zeros([size, img_w * img_h])
    Y = np.zeros([size, label_len * chars_len])
    left = (idx * size) % imgs_len
    right = (idx * size + size) % imgs_len
    img_bch = imgs[left: right] if right > left else imgs[left:] + imgs[:right]
    for i, img_name in enumerate(img_bch):
        label = img_name.split('_')[2][:label_len].upper()
        X[i:] = np.array(Image.open(os.path.join('std', img_name))).flatten() / 255
        for j, c in enumerate(label):
            try:
                Y[i, j * chars_len + chars.index(c)] = 1
            except Exception as e:
                print(img_name)
    return X, Y


def cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, img_h, img_w, 1])

    wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    wd1 = tf.get_variable(name='wd1', shape=[8 * 20 * 128, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    wout = tf.get_variable('name', shape=[1024, label_len * chars_len], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    bout = tf.Variable(b_alpha * tf.random_normal([label_len * chars_len]))
    out = tf.add(tf.matmul(dense, wout), bout)
    return out


def train():
    output = cnn()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    y_ = tf.reshape(output, [-1, label_len, chars_len])
    max_idx_p = tf.argmax(y_, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, label_len, chars_len]), 2)
    # correct_pred = tf.equal(max_idx_p, max_idx_l)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    correct_pred = tf.reduce_sum(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.int8), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_pred, 4), tf.int8))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        while True:
            batch_x, batch_y = get_batch(step, 100)
            _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_batch(step, 100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if acc > 0.5:
                    saver.save(sess, "model",)
                    break
            step += 1


def predict():
    output = cnn()
    saver = tf.train.Saver()
    count = 0
    with tf.Session() as sess:
        saver.restore(sess, "model")
        y_ = tf.argmax(tf.reshape(output, [-1, label_len, chars_len]), 2)
        for i, img_name in enumerate(test_imgs):
            label = img_name.split('_')[2][:label_len].upper()
            captcha_image = np.array(Image.open(os.path.join('std', img_name))).flatten() / 255
            text_list = sess.run(y_, feed_dict={X: [captcha_image], keep_prob: 1.})
            text = text_list[0].tolist()
            label_ = ''.join([chars[i] for i in text])
            if label == label_:
                count += 1
            print(label_, label)
        print(count/i)


def debug_cnn():
    output = cnn()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    y_ = tf.reshape(output, [-1, label_len, chars_len])
    max_idx_p = tf.argmax(y_, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, label_len, chars_len]), 2)
    correct_pred = tf.reduce_sum(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.int8), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_pred, 4), tf.int8))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_x, batch_y = get_batch(0, 10)
        _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
        batch_x_test, batch_y_test = get_batch(0, 10)
        p, l, c, a = sess.run([max_idx_p, max_idx_l, correct_pred, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        print(a)
        for i, v in enumerate(p):
            print(v.tolist(), l[i].tolist(), c[i])


if __name__ == '__main__':
    #reshape_and_gray()
    X = tf.placeholder(tf.float32, [None, img_h * img_w])
    Y = tf.placeholder(tf.float32, [None, label_len * chars_len])
    keep_prob = tf.placeholder(tf.float32)
    #train()
    predict()
    #debug_cnn()
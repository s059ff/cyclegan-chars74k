import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import cupy as xp
import datetime
import glob
import numpy as np
import os
import shutil
import tarfile
import urllib.request

from PIL import Image
from PIL import ImageOps

from model import Generator
from model import Discriminator
from visualize import visualize


# Define constants
N = 1000    # Minibatch size
M = 100000
SNAPSHOT_INTERVAL = 10
REAL_LABEL = 1
FAKE_LABEL = 0

def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/hnd.npy'):
        url = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz'
        response = urllib.request.urlopen(url)
        with open('dataset/EnglishHnd.tgz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/EnglishHnd.tgz') as stream:
            stream.extractall('dataset/')
        train = []
        for path in glob.iglob('dataset/English/Hnd/**/*.png', recursive=True):
            image = Image.open(path)
            image = ImageOps.grayscale(image)
            image = ImageOps.invert(image)
            image = image.resize((32, 32), Image.LINEAR)
            train.append(np.asarray(image))
        train = np.asarray(train, dtype='f') / 255.
        np.save('dataset/hnd', train)
    if not os.path.exists('dataset/fnt.npy'):
        url = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'
        response = urllib.request.urlopen(url)
        with open('dataset/EnglishFnt.tgz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/EnglishFnt.tgz') as stream:
            stream.extractall('dataset/')
        train = []
        for path in glob.iglob('dataset/English/Fnt/**/*.png', recursive=True):
            image = Image.open(path)
            image = ImageOps.grayscale(image)
            image = ImageOps.invert(image)
            image = image.resize((32, 32), Image.LINEAR)
            train.append(np.asarray(image))
        train = np.asarray(train, dtype='f') / 255.
        np.save('dataset/fnt', train)
    os.remove('dataset/EnglishHnd.tgz') if os.path.exists('dataset/EnglishHnd.tgz') else None
    os.remove('dataset/EnglishFnt.tgz') if os.path.exists('dataset/EnglishFnt.tgz') else None
    shutil.rmtree('dataset/English', ignore_errors=True)

    # Create samples.
    trainA = np.load('dataset/hnd.npy').astype('f')
    trainB = np.load('dataset/fnt.npy').astype('f')
    trainA = trainA.reshape((len(trainA), 1, 32, 32))
    trainB = trainB.reshape((len(trainB), 1, 32, 32))
    trainA = np.random.permutation(trainA)
    trainB = np.random.permutation(trainB)
    validationA = trainA[0:100]
    validationB = trainB[0:100]

    # (Align the number of data)
    _ = np.zeros((M, 1, 32, 32), dtype='f')
    for n in range(M):
        _[n] = trainA[n % len(trainA)]
    trainA = _
    _ = np.zeros((M, 1, 32, 32), dtype='f')
    for n in range(M):
        _[n] = trainB[n % len(trainB)]
    trainB = _

    # Create the model
    genA = Generator()      # genA convert B -> A
    genB = Generator()      # genB convert A -> B
    disA = Discriminator()  # disA discriminate realA and fakeA
    disB = Discriminator()  # disB discriminate realB and fakeB

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    genA.to_gpu()
    genB.to_gpu()
    disA.to_gpu()
    disB.to_gpu()

    # Setup optimizers
    optimizer_genA = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_genB = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_disA = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_disB = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_genA.setup(genA)
    optimizer_genB.setup(genB)
    optimizer_disA.setup(disA)
    optimizer_disB.setup(disB)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # (Validate input images)
    realA = validationA
    realB = validationB
    visualize(realA, 'realA.png')
    visualize(realB, 'realB.png')

    # Training
    for epoch in range(1000):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)
            realA = validationA
            realB = validationB
            fakeA = chainer.cuda.to_cpu(genA(chainer.cuda.to_gpu(realB)).data)
            fakeB = chainer.cuda.to_cpu(genB(chainer.cuda.to_gpu(realA)).data)
            visualize(fakeA, 'fakeA.png')
            visualize(fakeB, 'fakeB.png')
            chainer.serializers.save_hdf5("genA.h5", genA)
            chainer.serializers.save_hdf5("genB.h5", genB)
            chainer.serializers.save_hdf5("disA.h5", disA)
            chainer.serializers.save_hdf5("disB.h5", disB)
            os.chdir('..')

        total_loss_disA = 0.0
        total_loss_disB = 0.0
        total_loss_recA = 0.0
        total_loss_recB = 0.0
        total_loss_gen = 0.0

        for n in range(0, M, N):
            batchA = xp.array(trainA[n:n + N], dtype='f')
            batchB = xp.array(trainB[n:n + N], dtype='f')
            realA = batchA
            realB = batchB
            fakeA = genA(realB)
            fakeB = genB(realA)

            ############################
            # (1) Update D network
            ###########################
            # dis A
            y_realA = disA(realA)
            y_fakeA = disA(fakeA)
            loss_disA = (F.sum((y_realA - REAL_LABEL) ** 2) + F.sum((y_fakeA - FAKE_LABEL) ** 2)) / np.prod(y_fakeA.shape)

            # dis B
            y_realB = disB(realB)
            y_fakeB = disB(fakeB)
            loss_disB = (F.sum((y_realB - REAL_LABEL) ** 2) + F.sum((y_fakeB - FAKE_LABEL) ** 2)) / np.prod(y_fakeB.shape)

            # update dis
            disA.cleargrads()
            disB.cleargrads()
            loss_disA.backward()
            loss_disB.backward()
            optimizer_disA.update()
            optimizer_disB.update()

            ###########################
            # (2) Update G network
            ###########################

            # gen A
            fakeA = genA(realB)
            y_fakeA = disA(fakeA)
            loss_genA = F.sum((y_fakeA - REAL_LABEL) ** 2) / np.prod(y_fakeA.shape)

            # gen B
            fakeB = genB(realA)
            y_fakeB = disB(fakeB)
            loss_genB = F.sum((y_fakeB - REAL_LABEL) ** 2) / np.prod(y_fakeB.shape)

            # rec A
            recA = genA(fakeB)
            loss_recA = F.mean_absolute_error(recA, realA)

            # rec B
            recB = genB(fakeA)
            loss_recB = F.mean_absolute_error(recB, realB)

            # gen loss
            loss_gen = loss_genA + loss_genB + 10 * (loss_recA + loss_recB)

            # update gen
            genA.cleargrads()
            genB.cleargrads()
            loss_gen.backward()
            optimizer_genA.update()
            optimizer_genB.update()

            total_loss_disA += loss_disA.data
            total_loss_disB += loss_disB.data
            total_loss_recA += loss_recA.data
            total_loss_recB += loss_recB.data
            total_loss_gen += loss_gen.data

        # (View loss)
        total_loss_disA /= M / N
        total_loss_disB /= M / N
        total_loss_recA /= M / N
        total_loss_recB /= M / N
        total_loss_gen /= M / N
        print(epoch, total_loss_disA, total_loss_disB, total_loss_recA, total_loss_recB, total_loss_gen)


if __name__ == '__main__':
    main()

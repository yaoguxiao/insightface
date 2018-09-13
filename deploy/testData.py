import sys
import os
import mxnet as mx
import argparse
sys.path.append(os.path.join(os.getcwd(), "../src/common"))
sys.path.append(os.path.join(os.getcwd(), "../src/eval"))
import verification

def argParser():
    parser = argparse.ArgumentParser(description='test network')
    parser.add_argument('--model', default='../../insightface/models/model-res4-8-16-4-dim512/model,0', help='path of model')
    parser.add_argument('--data-dir', default='../../insightface/datasets/faces_ms1m_112x112/', help='path of test data')
    parser.add_argument('--target', default='lfw', help='name of test data')
    parser.add_argument('--output', default='fc1', help='output name')
    parser.add_argument('--batch-size', default=50, help='batch size')
    # parser.add_argument('add_argument')
    args = parser.parse_args()
    return args

def reaTestData():
    verList = {}
    for name in args.target.split(','):
        print("============", name)
        path = os.path.join(args.data_dir,name+".bin")
        print(path)
        if not os.path.exists(path):break
        verList[name] = verification.load_bin(path, [112,112])
        print('ver', name)
    return verList

def verTest(model, nbatch):
  results = []
  verList = reaTestData()
  print("===============, line:", sys._getframe().f_lineno)
  if verList is None:
      print("read test data err")
      return
  print("===============, line:", sys._getframe().f_lineno)
  for i in verList:
    print("===============, line:", sys._getframe().f_lineno)
    acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(verList[i], model, args.batch_size, 10, None, None)
    print('[%s][%d]XNorm: %f' % (i, nbatch, xnorm))
    # print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (i, nbatch, acc1, std1))
    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (i, nbatch, acc2, std2))
    results.append(acc2)
  return results

# class faceMode:
#     def __init__(self, args):
#         self.arts = args
#         modelid = args.model.split(',')
#         print(modelid[0], modelid[1])
#         sym, argParams, auxParams =  mx.model.load_checkpoint(modelid[0], int(modelid[1]))#type:mx.symbol.symbol.Symbol
#         sym = sym.get_internals()[args.output + '_output']
#         self.model = mx.mod.Module(symbol=sym, label_names=None)
#         self.model.bind(('data', (1, 3, 112,112)))
#         self.model.set_params(argParams, auxParams)
#         print(type(sym))

if __name__ == "__main__":
    args = argParser()
    # faceMode(args)

    modelid = args.model.split(',')
    print(modelid[0], modelid[1])
    sym, argParams, auxParams = mx.model.load_checkpoint(modelid[0], int(modelid[1]))  # type:mx.symbol.symbol.Symbol
    sym = sym.get_internals()[args.output + '_output']
    model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
    # model.bind(data_shapes=('data', (args.batch_size, 3, 112, 112)))
    model.bind(data_shapes=[('data', (args.batch_size, 3, 112,112))])
    model.set_params(argParams, auxParams)

    verTest(model, args.batch_size)
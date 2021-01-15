import argparse
import os
import pickle

version = 'Version 3.X => X.X'
print(" [*] Adjust compatibility powered by Tomosada")

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default='DCTGAN-Advanced', help='model architecture')
parser.add_argument("--train_name",type=str,default="Please Define youself")
parser.add_argument("--param_name",type=str,default="Please Define youself")
parser.add_argument("--data_folder",type=str,default="/home/tomosada/data")
parser.add_argument("--checkpoint",type=str,default="checkpoint")
parser.add_argument('--version', action='store_true')
parser.add_argument('--import_Github', action='store_true')
args = parser.parse_args()

train_name = args.model + '-' + args.train_name + '-' + args.param_name

os.makedirs(os.path.join(args.data_folder, args.model + '-P'), exist_ok=True)
os.chdir(os.path.join(args.data_folder, args.model + '-P'))
print(' [*] Current Directory', os.getcwd())

if args.version:
    print(' [*] You will adjust {} compatibility'.format(train_name))
    ans = input('Is this ok [y/N]: ')

    if ans == 'y':
        if os.path.exists(os.path.join(args.checkpoint, train_name, 'status.pickle')):
            print(" [*] Load SUCCESS", train_name)
            with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'rb') as f:
                global_epoch, global_step = pickle.load(f)
            print(" [*] Global Epoch: {0} Global Step: {1}".format(global_epoch, global_step))
        else:
            print(" [!] Load failed...", os.path.join(args.checkpoint, train_name, 'status.pickle'))
            exit()

        counter_tmp = input('Plese input counter: ')
        counter = int(counter_tmp)
        with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'wb') as f:
            pickle.dump([counter, global_epoch, global_step] , f)
    else:
        exit()

if args.import_Github:
    print(' [*] You will import model {} from Github'.format(train_name))
    ans = input('Is this ok [y/N]: ')

    if ans == 'y':
        with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'wb') as f:
            pickle.dump([9999, 999999, 999999] , f)
        print(' [*] Write Status SUCCEED')
        print(' [*] You need rename the weight to "XXXXXX_999999.pth"')


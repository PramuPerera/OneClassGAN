import options
import ocgantestdisjoint
import cvpr
import numpy as np
import random
import os
opt = options.test_options()

opt.istest = 0
#First use the validation set to pick best model:wq
text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
text_file1 = open(opt.dataset + "_progress1.txt", "w")
text_file1.close()
for itt in range(50):
        text_file1 = open(opt.dataset + "_progress1.txt", "a")        
	text_file = open(opt.dataset + "_progress.txt", "a")
	os.system('python2  cvprappend.py --epoch 501  --expname grapesip64 --img_wd 64  --img_ht 64  --depth 3  --datapath ../ --noisevar 0.2  --lambda1 0.4 ')

	auc1 = []
	auc2=[]
	auc3=[]
	auc4=[]
	ran = range(0,500,10)
	for i in ran:
	    opt.epochs = i
	    roc_auc = ocgantestdisjoint.main(opt)
	    print(roc_auc)
	    auc1.append(roc_auc[0])
	    auc2.append(roc_auc[1])
	    auc3.append(roc_auc[2])
	    auc4.append(roc_auc[3])

	#Pick best model w.r.t criterion 1
	i = np.argmin(np.array(auc1))
        imax = np.argmax(np.array(auc1))
	opt.epochs = ran[i]
	opt.istest=1
        res = ocgantestdisjoint.main(opt)[0]
        print(res)
        opt.epochs = ran[imax]
        opt.istest=1
        res1 = ocgantestdisjoint.main(opt)[0]

        text_file.write("%s \n" % (str(res)))
	text_file.close()

        text_file1.write("%s \n" % (str(res1)))
        text_file1.close()
'''
	print("AUC for criterion 1 (test): " + str(ocgantestdisjoint.main(opt)[0]))





#Pick best model w.r.t criterion 2
i = np.argmin(np.array(auc2))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 2 (val): " + str(ocgantestdisjoint.main(opt)[1]))
opt.istest=1
print("AUC for criterion 2 (test): " + str(ocgantestdisjoint.main(opt)[1]))



#Pick best model w.r.t criterion 3
i = np.argmin(np.array(auc3))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 3 (val): " + str(ocgantestdisjoint.main(opt)[2]))
opt.istest=1
print("AUC for criterion 3 (test): " + str(ocgantestdisjoint.main(opt)[2]))



#Pick best model w.r.t criterion 4
i = np.argmin(np.array(auc4))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 4 (val): " + str(ocgantestdisjoint.main(opt)[3]))
opt.istest=1
print("AUC for criterion 4 (test): " + str(ocgantestdisjoint.main(opt)[3])'''

import logging
import matplotlib.pyplot as plt


def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging


def draw_from_log(file,path):
    print(file)

    train_loss = []
    train_acc = []
    train_f1_macro = []
    train_auc_macro = []
    train_recall_macro = []

    val_loss = []
    val_acc = []
    val_f1_macro = []
    val_auc_macro = []
    val_recall_macro = []

    test_loss = []
    test_acc = []
    test_f1_macro = []
    test_auc_macro = []
    test_recall_macro = []

    x = []

    for count, line in enumerate(open(file, 'rU')):

        if count % 4 == 0:
            line = line.split(": ")
            test_acc.append(float(line[1].split(";")[0]))
            test_f1_macro.append(float(line[2].split(";")[0]))
            test_auc_macro.append(float(line[4].split(";")[0]))
            test_recall_macro.append(float(line[3].split(";")[0]))


        elif count % 4 == 1:
            line = line.split(": ")
            train_acc.append(float(line[1].split(";")[0]))
            train_f1_macro.append(float(line[2].split(";")[0]))
            train_auc_macro.append(float(line[4].split(";")[0]))
            train_recall_macro.append(float(line[3].split(";")[0]))

        elif count % 4 == 2:
            line = line.split(": ")
            val_acc.append(float(line[1].split(";")[0]))
            val_f1_macro.append(float(line[2].split(";")[0]))
            val_auc_macro.append(float(line[4].split(";")[0]))
            val_recall_macro.append(float(line[3].split(";")[0]))

        elif count % 4 == 3:
            line = line.split(": ")
            x.append(int(line[1].split(";")[0]))
            train_loss.append(float(line[2].split(";")[0]))
            val_loss.append(float(line[3].split(";")[0]))
            test_loss.append(float(line[4].split(";")[0]))

    #draw
    plt.figure(12)
    plt.subplot(121)
    plt.plot(x, train_loss, label="train_losses")
    plt.plot(x, val_loss, label="vals_losses")
    plt.plot(x, test_loss, label="test_losses")
    plt.plot(x, val_acc, label="vals_acc")
    plt.plot(x, train_acc, label="train_acc")
    plt.plot(x, test_acc, label="test_acc")
    plt.legend()
    plt.title("loss-acc")

    plt.subplot(122)
    plt.plot(x, train_f1_macro, label="train_f1_macro")
    plt.plot(x, val_f1_macro, label="val_f1_macro")
    plt.plot(x, test_f1_macro, label="test_f1_macro")
    plt.plot(x, train_auc_macro,label="train_auc_macro")

    plt.plot(x, val_auc_macro,label="val_auc_macro")

    plt.plot(x, test_auc_macro, label="test_auc_macro")

    plt.plot(x, test_recall_macro, label="test_recall_macro")


    plt.legend()
    plt.title("macro")
    plt.savefig(path)
    plt.show()


def draw_from_log_2(file,path):
    print(file)

    train_loss = []
    train_acc = []
    train_f1_macro = []
    train_auc_macro = []
    train_recall_macro = []
    test_loss = []
    test_acc = []
    test_f1_macro = []
    test_auc_macro = []
    test_recall_macro = []
    x = []
    for count, line in enumerate(open(file, 'rU')):

        if count % 3 == 0:
            line = line.split(": ")
            test_acc.append(float(line[1].split(";")[0]))
            test_f1_macro.append(float(line[2].split(";")[0]))
            test_auc_macro.append(float(line[4].split(";")[0]))
            test_recall_macro.append(float(line[3].split(";")[0]))


        elif count % 3 == 1:
            line = line.split(": ")
            train_acc.append(float(line[1].split(";")[0]))
            train_f1_macro.append(float(line[2].split(";")[0]))
            train_auc_macro.append(float(line[4].split(";")[0]))
            train_recall_macro.append(float(line[3].split(";")[0]))

        elif count % 3 == 2:
            line = line.split(": ")
            x.append(int(line[1].split(";")[0]))
            train_loss.append(float(line[2].split(";")[0]))
            test_loss.append(float(line[3].split(";")[0]))


    print("load from log over!")
    plt.figure(12)
    plt.subplot(121)
    plt.plot(x, train_loss, label="train_losses")
    plt.plot(x, test_loss, label="test_losses")
    plt.plot(x, train_acc, label="train_acc")
    plt.plot(x, test_acc, label="test_acc")
    plt.legend()
    plt.title("loss-acc")

    plt.subplot(122)
    plt.plot(x, train_f1_macro, label="train_f1_macro")
    plt.plot(x, test_f1_macro, label="test_f1_macro")
    plt.plot(x, train_auc_macro,label="train_auc_macro")
    plt.plot(x, test_auc_macro, label="test_auc_macro")
    plt.plot(x, test_recall_macro, label="test_recall_macro")



    plt.legend()
    plt.title("macro")
    plt.savefig(path)
    print("save picture over!")
    plt.show()
    return 0


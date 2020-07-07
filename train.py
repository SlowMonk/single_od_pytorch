from datasets import *
import torchvision
import time
import shutil
import matplotlib.pyplot as plt
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from utils import *


def main():

    batch_size= 14
    workers = 4
    num_epochs = 500
    dataset_num = 83
    input_size = 300
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_folder = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/'
    train_dataset = BalloonDataset(data_folder,split='train',dataset_num=dataset_num,input_size=input_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn= collate_fn,
        num_workers=workers,
        pin_memory=True)  # note that we're passing the collate function here

    #multi gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    num_classes= 2
    criterion =None
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    #Train model
    model.train()


    train(train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer
          ,num_epochs = num_epochs
          ,input_size=input_size)



def train(train_loader,model,criterion,optimizer,num_epochs,input_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loss_min = 0.9
    total_train_loss = []
    start_time = time.time()
    train_loss = []
    checkpoint_path = '/home/jake/PycharmProjects/balloon_detection/weights/fasterrcnn_resnet50_fpn.pth'
    best_model_path = '/home/jake/PycharmProjects/balloon_detection/weights/best_fasterrcnn_resnet50_fpn.pth'

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    images_temp = []
    targets_temp = []
    for epoch in range(num_epochs):
        for i, (images, targets, image_ids) in enumerate(train_loader):
            #print(i)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)

        if epoch >0:
            score = mean_average_precision(input_size=input_size)
            print('[A]Epoch[{}] loss[{}] MAP:[{}%]'.format(epoch,epoch_train_loss,score))
        else:
            print('[B]Epoch[{}] loss[{}] MAP:[{}%]'.format(epoch, epoch_train_loss, 0.00))
        checkpoint = {
            'epoch': epoch + 1,
            'train_loss_min': epoch_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        lr_scheduler.step()

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        ## TODO: save the model if validation loss has decreased
        if epoch_train_loss <= train_loss_min:
            print(
                'Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...to {}'.format(train_loss_min, epoch_train_loss,best_model_path),)
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            train_loss_min = epoch_train_loss

        time_elapsed = time.time() - start_time


def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def plot_img(data,idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0,255,0), thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
# helper functions to save best model

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def mean_average_precision(input_size):
    #input_size = 300
    batch_size = 13
    dataset_num = 83

    data_folder = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/'
    checkpoint_fpath = '/home/jake/PycharmProjects/balloon_detection/weights/best_fasterrcnn_resnet50_fpn.pth'
    valid_dataset = BalloonDataset(data_folder, split='test', dataset_num=dataset_num, input_size=input_size)
    weight_path = '/home/jake/PycharmProjects/balloon_detection/weights/best_fasterrcnn_resnet50_fpn.pth'

    csv_path = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/via_region_data_TEST_70.csv'
    path = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/TEST/'

    ##predict boxes
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    df = pd.read_csv(csv_path)
    df = process_bbox(df)
    image_num = 0

    image_unique = df['#filename'].unique()
    # print('image_unique->',image_unique)
    mean_precisions = []

    for img in image_unique:
        ##input image new_image_model
        # img_path = path + df['#filename'][image_num]
        img_path = path + img
        new_image_model = make_input_image(img_path, input_size=input_size)

        ##true boxes
        df[df['#filename'] == df['#filename'][image_num]]
        # input image model(new_image_model)
        image_num = df[df['#filename'] == img].index[0]
        true_boxes = make_true_boxes_new_scale(df, image_num, input_size=input_size)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

        outputs = model([new_image_model])
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        boxes_pred = outputs[0]['boxes']
        confidences = outputs[0]['scores']
        threshold = 0.7

        boxes_pred = boxes_pred[confidences > threshold]
        confidences = confidences[confidences > threshold]

        confidences = list(confidences)
        box_true_scale = make_true_boxes_new_scale(df, image_num, input_size=input_size)

        score = calculate_precision(boxes_true=box_true_scale, boxes_pred=boxes_pred, confidences=confidences,threshold=0.5)
        mean_precisions.append(score)
    return np.mean(mean_precisions)
if __name__=='__main__':
    main()
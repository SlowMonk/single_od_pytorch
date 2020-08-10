from datasets import *
import torchvision
import matplotlib.pyplot as plt
import os
import time
import shutil
from utils_ball import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def process_bbox(df):
    df['bbox'] = df['region_shape_attributes'].apply(lambda x: eval(x))
    df['x'] = df['bbox'].apply(lambda x: x['x'])
    df['y'] = df['bbox'].apply(lambda x: x['y'])
    df['w'] = df['bbox'].apply(lambda x: x['width'])
    df['h'] = df['bbox'].apply(lambda x: x['height'])
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df['w'] = df['w'].astype(np.float)
    df['h'] = df['h'].astype(np.float)

    df.drop(columns=['bbox'], inplace=True)
    #     df.reset_index(drop=True)
    return df
def main():
    input_size =300
    batch_size = 13
    dataset_num = 83

    data_folder = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/'
    checkpoint_fpath = '/home/jake/PycharmProjects/balloon_detection/weights/best_fasterrcnn_resnet50_fpn.pth'
    valid_dataset = BalloonDataset(data_folder,split='test',dataset_num=dataset_num,input_size=input_size)
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



    image_unique=df['#filename'].unique()
    #print('image_unique->',image_unique)
    mean_precisions = []

    for img in image_unique:
        ##input image new_image_model
        #img_path = path + df['#filename'][image_num]
        img_path = path + img
        #print('===============================================================')
        #print('img_path->',img_path)
        new_image_model = make_input_image(img_path,input_size=input_size)

        ##true boxes
        df[df['#filename']==df['#filename'][image_num]]
        #input image model(new_image_model)
        image_num = df[df['#filename']==img].index[0]
        true_boxes = make_true_boxes_new_scale(df,image_num,input_size=input_size)


        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        #model,_,_,_=load_ckp(checkpoint_fpath,model,optimizer)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

        outputs = model([new_image_model])
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        boxes_pred = outputs[0]['boxes']
        confidences = list(outputs[0]['scores'])
        #print('outputs->',outputs)
        #print('pred_boxes->', boxes_pred)
        #print('confidences->',confidences)
        #print('true_boxes->',true_boxes)

        box_true_scale = make_true_boxes_new_scale(df, image_num,input_size=input_size)

        score = calculate_precision(boxes_true=box_true_scale, boxes_pred=boxes_pred, confidences=confidences, threshold=.5)
        mean_precisions.append(score)
    #print('calculate_precision:',np.mean(mean_precisions),'%')
    test_calc_precision()

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
    valid_loss_min = checkpoint['train_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


if __name__ == '__main__':
    main()
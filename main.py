import cv2
import copy
import numpy as np 
import torch
import segmentation_models_pytorch as smp
from sklearn.cluster import KMeans

size_height =437
size_width = 640#変更予定
HEIGHT = 256
WIDTH = 256

#haarcascades読み込み
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')

class face_app:
    def __init__(self):
        self.ENCODER = 'resnet34'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = ['background','skin']
        self.ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
        self.DEVICE = 'cpu'
        self.DECODER = 'unet'
        self.model = smp.Unet(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=len(self.CLASSES),
            activation=self.ACTIVATION,
        )
        self.model = self.model.to("cpu")
        self.model.load_state_dict(torch.load("models/model.pth",map_location=torch.device('cpu')))
        self.model.eval()        
        
    #顔検出
    def face_detect(self,input):
        #print(str(input.read(),'shift_jis'))
        input_image = cv2.imdecode(input, cv2.IMREAD_COLOR)
        #input_image = cv2.imread(input_image)
        faces = face_cascade.detectMultiScale(input_image, 1.1,3)
        for (x,y,w,h) in faces:
            # print(w,h)
            #face = cv2.rectangle(input_image,(x,y),(x+w,y+h),(1,1,1),2)
            face = input_image[y-50:y+h+100, x-50:x+w+100]
            face = cv2.resize(face,(HEIGHT,WIDTH))
        cv2.imwrite('a.png',face)
        return face

    #skinマスク抽出
    def make_mask(self,input):
        face = input
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)
        # 前処理
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        processed_image = preprocessing_fn(face)
        processed_image = processed_image.transpose(2, 0, 1).astype('float32')
        # モデルで推論
        torch_image = torch.from_numpy(processed_image).to(self.DEVICE).unsqueeze(0)
        predict = self.model(torch_image)
        predict = predict.detach().cpu().numpy()[0][0].reshape((HEIGHT,WIDTH))
        # 0.5以上を1とする
        predict_img = np.zeros([256,256]).astype(np.int8)
        predict_img = np.where(predict>0.5, 1 , predict_img)
        for x in range(HEIGHT):
            for y in range(WIDTH):
                if predict_img[x,y]==0:
                    face[x,y]=0,0,0
        face = cv2.resize(face,(512,512))
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_ex = copy.copy(face)
        cv2.imwrite('b.png',face)
        return face
        #pilImg = Image.fromarray(np.uint8(face))
        #pilImg.save('hoge.png')

    #目、口検出
    def part_detect(self,input):
        face = input
        eyes = eye_cascade.detectMultiScale(face,1.1, 30,cv2.CASCADE_FIND_BIGGEST_OBJECT)
        mouth = mouth_cascade.detectMultiScale(face,1.1, 100, cv2.CASCADE_FIND_BIGGEST_OBJECT)
        return eyes,mouth
    
    """
    #ニキビ検出
    def acne_detect(self,input):
        face_ex = input
        
        cv2.imwrite('face.png', face_ex)
        img_Lab = cv2.cvtColor(face_ex, cv2.COLOR_RGB2Lab)
        cv2.imwrite('lab.png', img_Lab)
        img_L, img_a, img_b = cv2.split(img_Lab)
        '''
        colormap = plt.get_cmap('bwr')
        heatmap = (colormap(img_a) * 2**8).astype(np.uint16)[:,:,:3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        '''
        img_Lab = cv2.merge((img_L,img_a,img_b))
        return img_Lab
    """
    #シワ検出    
    def wrinkle_detect(self,input):
        eyes,mouth = self.part_detect(input)
        face = input
        face_wr = copy.copy(face)
        threshold1 = 60
        threshold2 = 50
        fx = 0
        fy = 0
        fw = 0
        fh = 0
        #目元
        for i,(ex,ey,ew,eh) in enumerate(eyes):
            fx += (ex+ew/2)/2
            fy += (ey-eh)/2
            fw += ew
            fh += eh/2
            eye = face[ey+int(0.5*eh):ey+int(1.5*eh),ex-int(0.8*ew):ex+2*ew]
            edge_img_e = cv2.Canny(eye, threshold1, threshold2)
            edge_img_e = cv2.cvtColor(edge_img_e, cv2.COLOR_GRAY2RGB)
            edge_e_b = cv2.addWeighted(src1=face[ey+int(0.5*eh):ey+int(1.5*eh),ex-int(0.8*ew):ex+2*ew],alpha=0.8,src2=edge_img_e,beta=0.2,gamma=0)
            face_wr[ey+int(0.5*eh):ey+int(1.5*eh),ex-int(0.8*ew):ex+2*ew] = edge_e_b
        print(eyes)
        print(mouth)
        yl = int(fy-fh)
        yh = int(fy+fh)
        xl = int(fx-fw)
        xr = int(fx+fw)
        if int(fy-fh)<0 :
            yl = 0
        if int(fy+fh)>512:
            yh = 512
        if int(fx-fw)<0:
            xl = 0
        if int(fx+fw)>512:
            xr = 0
        #ひたい
        forehead = face[yl:yh,xl:xr]
        forehead = cv2.cvtColor(forehead, cv2.COLOR_RGB2GRAY)
        edge_img_f = cv2.Canny(forehead, threshold1, threshold2)
        edge_img_f = cv2.cvtColor(edge_img_f, cv2.COLOR_GRAY2RGB)
        edge_f_b = cv2.addWeighted(src1=face[yl:yh,xl:xr],alpha=0.8,src2=edge_img_f,beta=0.2,gamma=0)
        face_wr[yl:yh,xl:xr] = edge_f_b
        #口周り
        for (mx,my,mw,mh) in mouth:
            yl2 = int(my-0.8*mh)
            yh2 =int(my+1.5*mh)
            xl2 = mx-mw
            xr2 = mx+2*mw
            if yl2 < 0:
                yl2 = 0
            if yh2>512:
                yh2 = 512
            if xl2 <0:
                xl2 = 0
            if xr2 > 512 :
                xr2 = 0
            mouth_around = face[yl2:yh2,xl2:xr2]
            edge_img_m = cv2.Canny(mouth_around, threshold1, threshold2)
            edge_img_m = cv2.cvtColor(edge_img_m, cv2.COLOR_GRAY2RGB)
            edge_m_b = cv2.addWeighted(src1=face[yl2:yh2,xl2:xr2],alpha=0.8,src2=edge_img_m,beta=0.2,gamma=0)
            face_wr[yl2:yh2,xl2:xr2] = edge_m_b
        return face_wr
        #cv2.imwrite('sample_blend.jpg', face_wr)
        
    #肌色検出   
    def skin_color(self,input):
        face = input
        face_skc = copy.copy(face)
        face_skc = cv2.cvtColor(face_skc,cv2.COLOR_BGR2RGB)
        face_skc = face_skc.reshape((face_skc.shape[0] * face_skc.shape[1], 3))
        cluster = KMeans(n_clusters=2)
        cluster.fit(face_skc)
        KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
            n_clusters=5, n_init=10,
            random_state=None, tol=0.0001, verbose=0)
        cluster_centers_arr = cluster.cluster_centers_.astype(int, copy=False)
        for i,ar in enumerate(cluster_centers_arr):
            print(ar)
            count = 0
            for a in ar:
                if a<25:
                    count+=1
            if count == 3:
                print(i,ar)
                cluster_centers_arr = np.delete(cluster_centers_arr,i,0)

        for i, rgb_arr in enumerate(cluster_centers_arr):
            color_hex_str = '#%02x%02x%02x' % tuple(rgb_arr)
        #tiled_color_img.show()
        return color_hex_str

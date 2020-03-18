class yolo():
    def __init__(self,IMAGE_W=416,IMAGE_W=416,GRID_W=13,GRID_H=13,anc_box=False,BOX=5):
        
       self.IMAGE_W=IMAGE_W
       self.IMAGE_H=IMAGE_H
       self.GRID_W=GRID_W
       self.GRID_H=GRID_H
       self.CLASS=CLASS
       self.BOX=BOX
       self.anc_box=anc_box
       self._grid_offset=grid_offset
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(self):
        return tf.space_to_depth(x, block_size=2)

    def build_model(self):
      input_image = Input(shape=(self.IMAGE_H, self.IMAGE_W, 3))

      # Layer 1
      x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)

      x = BatchNormalization(name='norm_1')(x)
      x = LeakyReLU(alpha=0.1)(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

      # Layer 2
      x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
      x = BatchNormalization(name='norm_2')(x)
      x = LeakyReLU(alpha=0.1)(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

      # Layer 3
      x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
      x = BatchNormalization(name='norm_3')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 4
      x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
      x = BatchNormalization(name='norm_4')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 5
      x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
      x = BatchNormalization(name='norm_5')(x)
      x = LeakyReLU(alpha=0.1)(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

      # Layer 6
      x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
      x = BatchNormalization(name='norm_6')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 7
      x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
      x = BatchNormalization(name='norm_7')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 8
      x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
      x = BatchNormalization(name='norm_8')(x)
      x = LeakyReLU(alpha=0.1)(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

      # Layer 9
      x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
      x = BatchNormalization(name='norm_9')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 10
      x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
      x = BatchNormalization(name='norm_10')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 11
      x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
      x = BatchNormalization(name='norm_11')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 12
      x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
      x = BatchNormalization(name='norm_12')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 13
      x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
      x = BatchNormalization(name='norm_13')(x)
      x = LeakyReLU(alpha=0.1)(x)

      skip_connection = x

      x = MaxPooling2D(pool_size=(2, 2))(x)

      # Layer 14
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
      x = BatchNormalization(name='norm_14')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 15
      x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
      x = BatchNormalization(name='norm_15')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 16
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
      x = BatchNormalization(name='norm_16')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 17
      x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
      x = BatchNormalization(name='norm_17')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 18
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
      x = BatchNormalization(name='norm_18')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 19
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
      x = BatchNormalization(name='norm_19')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 20
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
      x = BatchNormalization(name='norm_20')(x)
      x = LeakyReLU(alpha=0.1)(x)

      # Layer 21
      skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
      skip_connection = BatchNormalization(name='norm_21')(skip_connection)
      skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
      skip_connection = Lambda(self.space_to_depth_x2)(skip_connection)

      x = concatenate([skip_connection, x])

      # Layer 22
      x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
      x = BatchNormalization(name='norm_22')(x)
      x = LeakyReLU(alpha=0.1)(x)

      if self.anc_box==True:
        # Layer 23
        x = Conv2D(self.BOX*(4 + 1 + self.CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
        output = Reshape((self.GRID_H, self.GRID_W,self.BOX,4 + 1 + self.CLASS))(x)
      else :
        # Layer 23
        x = Conv2D((4 + 1 + self.CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
        output = Reshape((self.GRID_H, self.GRID_W,4 + 1 + self.CLASS))(x)

      # small hack to allow true_boxes to be registered when Keras build the model 
      # for more information: https://github.com/fchollet/keras/issues/2790
      #output = Lambda(lambda args: args[0])([output, true_boxes])#Change :Hasib

      #model = Model([input_image, true_boxes], output)#Change :Hasib
      model = Model(input_image, output)
      return model
      #model.load_weights('/content/drive/My Drive/Data/yolo_net_ep500_act.h5')


    def yolo_loss_1(self,y_true, y_pred):
          loss=0
              ### adjust w and h
          obj_mask_ex= tf.expand_dims(y_true[..., 4], axis=-1)
          obj_mask= y_true[..., 4]
          #conf_obj_mask=y_true[...,4]
          noobj_mask=np.abs(y_true[...,4]-1)

          if self.anc_box==True and self._grid_offset==True:
            _x = tf.to_float(tf.reshape(tf.tile(tf.range(self.GRID_W), [self.GRID_H]), (1, self.GRID_H, self.GRID_W, 1, 1)))#1,13,13,1,1
            _y = tf.transpose(_x, (0,2,1,3,4))#1,13,13,1,1
            _grid = tf.tile(tf.concat([_x,_y], -1), [BATCH_SIZE, 1, 1, self.BOX, 1])#10,13,13,5,1

            pred_xy = tf.sigmoid(y_pred[..., :2]) + _grid
            pred_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,self.BOX,2])

          if self.anc_box==False and self._grid_offset==True:
            _x = tf.to_float(tf.reshape(tf.tile(tf.range(self.GRID_W), [self.GRID_H]), (1, self.GRID_H, self.GRID_W, 1)))#1,13,13,1,1
            _y = tf.transpose(_x, (0,2,1,3))#1,13,13,1
            _grid = tf.tile(tf.concat([_x,_y], -1), [BATCH_SIZE, 1, 1, 1])#10,13,13,1

            pred_xy = tf.sigmoid(y_pred[..., :2]) + _grid
            pred_wh = y_pred[..., 2:4]# * np.reshape(ANCHORS, [1,1,1,self.BOX,2])
            #pred_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,self.BOX,2])

          if self._grid_offset==False:
            pred_xy=y_pred[...,0:2] #+ cell_grid # if cell_grid Batch_Gen center_x -=grid_x
            pred_wh=y_pred[...,2:4]
          # compute grid factor and net factor
          grid_factor = tf.reshape(tf.cast([self.GRID_W, self.GRID_H], tf.float32), [1,1,1,1,2])   
          #net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])

          true_xy=y_true[...,0:2]/grid_factor
          true_wh=y_true[...,2:4]/grid_factor
          pred_xy/=grid_factor
          pred_wh/=grid_factor
          pred_conf=tf.sigmoid(y_pred[...,4])
          ### adjust confidence
          true_wh_half = true_wh / 2.
          true_mins    = tf.subtract(true_xy,true_wh_half)
          true_maxes   = tf.add(true_xy,true_wh_half)

          pred_wh_half = pred_wh / 2.
          pred_mins    = tf.subtract(pred_xy,pred_wh_half)
          pred_maxes   = tf.add(pred_xy,pred_wh_half)       

          intersect_mins  = tf.maximum(pred_mins,  true_mins)
          intersect_maxes = tf.minimum(pred_maxes, true_maxes)
          intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
          intersect_areas = tf.multiply(intersect_wh[..., 0] , intersect_wh[..., 1])

          true_areas = tf.multiply(true_wh[..., 0] , true_wh[..., 1])
          pred_areas = tf.multiply(pred_wh[..., 0] , pred_wh[..., 1])

          union_areas =tf.subtract(tf.add(pred_areas,true_areas),intersect_areas)
          intersect_areas=tf.add(intersect_areas,1)
          union_areas=tf.add(union_areas,1)
          iou_scores  = tf.truediv(intersect_areas, union_areas)
          true_box_class = tf.argmax(y_true[..., 5:], -1)

          pred_box_class = y_pred[..., 5:]
          class_mask = y_true[..., 4] * tf.gather(self.CLASS_WEIGHTS, true_box_class)

          #class_mask = y_true[..., 4] * tf.to_float(true_box_class)
          nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
          true_conf =tf.multiply( iou_scores,y_true[..., 4])

          loss_bb=tf.subtract(true_xy,pred_xy)
          loss_bb=tf.square(loss_bb)
          loss_bb=tf.multiply(loss_bb,obj_mask_ex)
          loss_bb=tf.reduce_sum(loss_bb)

          pred_wh_abs=tf.abs(pred_wh)
          pred_wh_abs=tf.where(tf.equal(pred_wh_abs,0),tf.ones_like(pred_wh_abs),pred_wh_abs)
          pred_wh_sign=tf.truediv(pred_wh,pred_wh_abs)
          loss_wh=tf.subtract(tf.sqrt(true_wh),tf.multiply(pred_wh_sign,tf.sqrt(pred_wh_abs)))
          #loss_wh=tf.subtract(true_wh,pred_wh)
          loss_wh=tf.square(loss_wh)
          loss_wh=tf.multiply(loss_wh,obj_mask_ex)
          loss_wh=tf.reduce_sum(loss_wh)

          loss_conf=tf.subtract(true_conf,pred_conf)
          loss_conf=tf.square(loss_conf)
          loss_conf=tf.multiply(loss_conf,obj_mask)
          loss_conf=tf.reduce_sum(loss_conf)

          loss_noobj_conf=tf.subtract(true_conf,pred_conf)
          loss_noobj_conf=tf.square(loss_noobj_conf)
          loss_noobj_conf=tf.multiply(loss_noobj_conf,noobj_mask)
          loss_noobj_conf=tf.reduce_sum(loss_noobj_conf)


          true_box_class=tf.cast(true_box_class,tf.int32)
          loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
          loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

          loss=lambda_coord*loss_bb+lambda_coord*loss_wh+loss_conf+lambda_noobj*loss_noobj_conf+loss_class  
          return loss
        
    def IOU(self,bboxes1, bboxes2):
        #import pdb;pdb.set_trace()
        x1_min, y1_min, x1_max, y1_max = list(bboxes1)
        x2_min, y2_min, x2_max, y2_max = list(bboxes2)
        xA = np.maximum(x1_min, x2_min)
        yA = np.maximum(y1_min, y2_min)
        xB = np.minimum(x1_max, x2_max)
        yB = np.minimum(y1_max, y2_max)
        interArea = np.maximum((xB - xA ), 0) * np.maximum((yB - yA ), 0)
        boxAArea = (x1_max - x1_min ) * (y1_max - y1_min )
        boxBArea = (x2_max - x2_min ) * (y2_max - y2_min )
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou
    
    def Batch_Gen(self,all_data,no_of_batch,_dir):
          while(True):
                N=len(all_data)
                _batch_size=N//no_of_batch

                for _ind in range(no_of_batch):

                    batch=all_data[_ind*_batch_size:(_ind+1)*_batch_size]
                    n=len(batch)
                    x_batch = np.zeros((n,self.IMAGE_H, self.IMAGE_W,3),dtype=np.float32)                         # input images
                    if self.anc_box==True:
                      y_batch = np.zeros((n, self.GRID_H, self.GRID_W,BOX,4+1+len(LABELS)),dtype=np.float)                # desired network output
                    else :
                      y_batch = np.zeros((n, self.GRID_H, self.GRID_W,4+1+len(LABELS)),dtype=np.float)                # desired network output
                    instance_count=0


                    for sample in batch:
                            self.IMAGE_name = sample['filename']
                            img = cv2.imread(root+_dir+self.IMAGE_name)
                            img = cv2.resize(img, (self.IMAGE_H,self.IMAGE_W))
                            img = img[:,:,::-1]
                            img_w=sample['height']
                            img_h=sample['width']
                            all_objs = sample['object']
                            # construct output from object's x, y, w, h
                            true_box_index = 0
                            anchors = [[0, 0, ANCHORS[2*i], ANCHORS[2*i+1]] for i in range(int(len(ANCHORS)//2))]
                            for obj in all_objs:

                                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in LABELS:
                                    center_x = .5*(obj['xmin'] + obj['xmax'])
                                    center_x = center_x*self.GRID_W
                                    center_y = .5*(obj['ymin'] + obj['ymax'])
                                    center_y=center_y*self.GRID_H
                                    self.GRID_x = int(np.floor(center_x))
                                    self.GRID_y = int(np.floor(center_y))
                                    if _self.GRID_offset==False:
                                                center_x-=self.GRID_x
                                                center_y-=self.GRID_y
                                    #center_x-=self.GRID_x
                                    #center_y-=self.GRID_y
                                    if self.GRID_x < self.GRID_W and self.GRID_y < self.GRID_H:
                                        obj_indx  = LABELS.index(obj['name'])
                                        center_w = (obj['xmax'] - obj['xmin'])*self.GRID_W #/ (float(self.config['self.IMAGE_W'])# / self.config['self.GRID_W']) # unit: grid cell
                                        center_h = (obj['ymax'] - obj['ymin'])*self.GRID_H #/ (float(self.config['self.IMAGE_H'])# / self.config['self.GRID_H']) # unit: grid cell
                                        center_w=center_w
                                        center_h=center_h

                                        box = [center_x, center_y, center_w, center_h]

                                        # find the anchor that best predicts this box#Change :Hasib
                                        best_anchor = -1
                                        max_iou     = -1

                                        shifted_box = [0, 0, center_w, center_h]

                                        for i in range(len(anchors)):
                                            anchor = anchors[i]
                                            iou    = self.IOU(shifted_box, anchor)

                                            if max_iou < iou:
                                                best_anchor = i
                                                max_iou     = iou

                                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                                        if self.anc_box==True:
                                          y_batch[instance_count, self.GRID_y, self.GRID_x, best_anchor,0:4] = box
                                          y_batch[instance_count, self.GRID_y, self.GRID_x, best_anchor,4  ] = 1.
                                          y_batch[instance_count, self.GRID_y, self.GRID_x, best_anchor,5+obj_indx] = 1
                                        else :
                                          y_batch[instance_count, self.GRID_y, self.GRID_x,0:4] = box
                                          y_batch[instance_count, self.GRID_y, self.GRID_x,4  ] = 1.
                                          y_batch[instance_count, self.GRID_y, self.GRID_x,5+obj_indx] = 1

                                        # assign the true box to b_batch
                                        #b_batch[instance_count, 0, 0, 0, true_box_index] = box#Change: Hasib

                                        #true_box_index += 1
                                        #true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

                            # assign input image to x_batch
                            x_batch[instance_count] = img/255

                            # increase instance counter in current batch
                            instance_count += 1  

                            #print(' new batch created', idx)
                    yield (x_batch, y_batch)


    def _softmax(self,x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)

    def _sigmoid(self,x):
        return 1. / (1. + np.exp(-x))

    def nmax_supp(self,boxes):
        df=pd.DataFrame(boxes,columns=['x','y','w','h','conf','_class'])
        sdf=df.sort_values('conf',ascending=False).reset_index(drop=True)
        #print(sdf)
        for ind1 in range(len(sdf)-1):
            box1=sdf.loc[ind1,['x','y','w','h']].values
            b1_class=sdf.loc[ind1,['_class']].values
            for ind2 in range(ind1+1,len(sdf)):
                b2_class=sdf.loc[ind2,['_class']].values
                b2_conf=sdf.loc[ind2,['conf']].values
                if b2_class==b1_class and b2_conf>0.0:
                     box2=sdf.loc[ind2,['x','y','w','h']].values
                     if b1_class==b2_class and IOU(box1,box2)>=0.1:
                        sdf.loc[ind2,'conf']=0.0
        ndf=sdf[sdf.conf>0.0]
        return list(ndf.values)#[:len(ndf.values)//2]

    def netout_to_box_anc(self,netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
        #self.GIRD_h, self.GIRD_w, nb_box = netout.shape[:3]
        self.GIRD_H, self.GIRD_W = netout.shape[:2]

        boxes = []
        # decode the output by the network
        netout[..., 4]  = self._sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * self._softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold

        for row in range(self.GIRD_H):
            for col in range(self.GIRD_W):
                for b in range(BOX):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    #classes = netout[row,col,5:]
                    confidence = netout[row,col,b,4]
                    if np.sum(classes) > 0:
                        x, y, w, h = netout[row,col,b,:4]
                        x = (col + _sigmoid(x)) / self.GIRD_w # center position, unit: image width
                        y = (row + _sigmoid(y)) / self.GIRD_h # center position, unit: image height
                        w = ANCHORS[2 * b + 0] * np.exp(w) / self.GIRD_w # unit: image width
                        h = ANCHORS[2 * b + 1] * np.exp(h) / self.GIRD_h # unit: image height
                        #print(x,y,w,h)
                        classes=np.argmax(classes)

                        box = (x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                        box = (x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                        boxes.append(box)   
        f_boxes=self.nmax_supp(boxes)
        return f_boxes    

    def netout_to_box(self,netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
        #self.GIRD_h, self.GIRD_w, nb_box = netout.shape[:3]
        boxes = []

        # decode the output by the network
        netout[..., 4]  = self._sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * self._softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold
        class_prob_log=[0]*self.CLASS
        #boxes=[(0,)*6]*self.CLASS
        boxes=[]
        for row in range(self.GIRD_H):
            for col in range(self.GIRD_W):
                #for b in range(BOX):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,5:]
                    #classes = netout[row,col,5:]
                    confidence = netout[row,col,4]
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        #x, y, w, h = netout[row,col,b,:4]
                        x, y, w, h = netout[row,col,:4]
                        #print(col,_sigmoid(x-col),row,_sigmoid(y),w,h)
                        w_ratio=(self.IMAGE_W/self.GIRD_W)/self.IMAGE_W
                        h_ratio=(self.IMAGE_W/self.GIRD_W)/self.IMAGE_H
                        x = (col+_sigmoid(x))*(h_ratio)#x*(32/416) # center position, unit: image width
                        y = (row+_sigmoid(y))*(w_ratio)#y*(32/416) # center position, unit: image height
                        w = w*(w_ratio) # unit: image width
                        h = h*(h_ratio) # unit: image height
                        #print(x,y,w,h)
                        class_ind=np.argmax(classes)
                        #if class_prob_log[class_ind]< classes[class_ind]:       
                        class_prob_log[class_ind]=classes[class_ind]
                        box = (x-w/2, y-h/2, x+w/2, y+h/2, confidence, class_ind)
                        #if abs(box[0])<=1 and abs(box[1])<=1 and box[2]<=1 and box[3]<=1 :
                        #    if (box[0])>=0 and (box[1])>=0 and box[2]>=0 and box[3]>=0 :
                        boxes.append(box)
                                           #boxes[class_ind]=box



        f_boxes=self.nmax_supp(boxes)
        return f_boxes#[:5]   

    def draw_boxes(self,image, boxes, labels,t_lbl=None):
        image_w, image_h, _ = image.shape
        wf=image_w#/self.IMAGE_W
        hf=image_h#/self.IMAGE_H
        show_gt=False
        if show_gt==True:
            for box in t_lbl:
                box=list(box.values())
                xmin = int(box[1]*wf)
                ymin = int(box[2]*hf)
                xmax = int(box[3]*wf)
                ymax = int(box[4]*hf)

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
                cv2.putText(image, 
                            box[0] + ' ', 
                            (xmin, ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.0015 * 400, 
                            (255,0,0), 2)
        for box in boxes:
            xmin = int(box[0]*wf)
            ymin = int(box[1]*hf)
            xmax = int(box[2]*wf)
            ymax = int(box[3]*hf)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
            cv2.putText(image, 
                        labels[int(box[5])] + ' ' + str(box[4]), 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.0015 * 400, 
                        (0,255,0), 2)

        return image


class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')

        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset-size:self.offset]

        def reset(self):
            self.offset = 4

def load_weights(model,path_wst):
      wt_path = path_wts                      
      weight_reader = WeightReader(wt_path)
      weight_reader.reset()
      nb_conv = 19

      for i in range(1, nb_conv+1):
          conv_layer = model.get_layer('conv_' + str(i))

          if i < nb_conv:
              norm_layer = model.get_layer('norm_' + str(i))

              size = np.prod(norm_layer.get_weights()[0].shape)

              beta  = weight_reader.read_bytes(size)
              gamma = weight_reader.read_bytes(size)
              mean  = weight_reader.read_bytes(size)
              var   = weight_reader.read_bytes(size)

              weights = norm_layer.set_weights([gamma, beta, mean, var])       

          if len(conv_layer.get_weights()) > 1:
              bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
              kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
              kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
              kernel = kernel.transpose([2,3,1,0])
              conv_layer.set_weights([kernel, bias])
          else:
              kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
              kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
              kernel = kernel.transpose([2,3,1,0])
              conv_layer.set_weights([kernel])
      return model


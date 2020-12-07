yoloV3训练脚本的解析

* 首先定义一些重要的op

  ```python
          with tf.name_scope("define_loss"):
              self.model = YOLOV3(self.input_data, self.trainable)
              self.net_var = tf.global_variables()
              self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                  self.label_sbbox, self.label_mbbox, self.label_lbbox,
                  self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
              self.loss = self.giou_loss + self.conf_loss + self.prob_loss
  
          with tf.name_scope('learn_rate'):
              self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
              warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                         dtype=tf.float64, name='warmup_steps')
              train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
              self.learn_rate = tf.cond(
                  pred=self.global_step < warmup_steps,
                  true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                  false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                   (1 + tf.cos(
                                       (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
              )
              global_step_update = tf.assign_add(self.global_step, 1.0)
  
          with tf.name_scope("define_weight_decay"):
              moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
  
  ```

* 以上可统称为没卵用

  ```Python
      #这一步是用来搜集所有变量名，只要所主干网络节点的变量名，    
      with tf.name_scope("define_first_stage_train"):
              self.first_stage_trainable_var_list = []
              for var in tf.trainable_variables():
                  var_name = var.op.name
                  var_name_mess = str(var_name).split('/')
                  if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                      self.first_stage_trainable_var_list.append(var)
  
              first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                       var_list=self.first_stage_trainable_var_list)
              #一个上下文管理器，依次运行下面list节点
              with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                  with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                      with tf.control_dependencies([moving_ave]):
                          self.train_op_with_frozen_variables = tf.no_op()#什么都不做，仅做为点位符使用控制边界。但好像会拖慢速度
        
      #二阶段就简单的多，所有网络中的变量都将参与计算，同样适用上下文管理器控制计算流程
      with tf.name_scope("define_second_stage_train"):
          second_stage_trainable_var_list = tf.trainable_variables()
          second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                    var_list=second_stage_trainable_var_list)
  
          with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
              with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                  with tf.control_dependencies([moving_ave]):
                      self.train_op_with_all_variables = tf.no_op()
                      
                      
                      
             #后来
           train_op = self.train_op_with_all_variables
           #最终将运用到run当中做train_op
                      for train_data in pbar:
                  _, summary, train_step_loss, global_step_val = self.sess.run(
                      [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                          self.input_data: train_data[0],
                          self.label_sbbox: train_data[1],
                          self.label_mbbox: train_data[2],
                          self.label_lbbox: train_data[3],
                          self.true_sbboxes: train_data[4],
                          self.true_mbboxes: train_data[5],
                          self.true_lbboxes: train_data[6],
                          self.trainable: True,
                      }）
  ```

  

* 可视化模块

  ```python
          with tf.name_scope('summary'):
              tf.summary.scalar("learn_rate", self.learn_rate)
              tf.summary.scalar("giou_loss", self.giou_loss)
              tf.summary.scalar("conf_loss", self.conf_loss)
              tf.summary.scalar("prob_loss", self.prob_loss)
              tf.summary.scalar("total_loss", self.loss)
  
              # 训练过程可视化
  
              draw_box_result = tf.py_func(vis_func, [self.input_data,
                                                      self.model.pred_sbbox,  # [b,52,52,3,5+cls_num]
                                                      self.model.pred_mbbox,
                                                      self.model.pred_lbbox,
                                                      self.num_classes
                                                      ], [tf.uint8], name='train_draw_box')
              tf.summary.image('show_train_process_image', draw_box_result[0], max_outputs=6)
              logdir = "./data/log/"
              if os.path.exists(logdir): shutil.rmtree(logdir)
              os.mkdir(logdir)
              self.write_op = tf.summary.merge_all()
              self.summary_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)
  
  ```

  
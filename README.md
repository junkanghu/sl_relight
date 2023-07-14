1. 非pretraine时，如果要fine tune albedo，默认使用albedo_vgg和albedo_gan。此时--vgg或--vgg_api只对shad有用。
2. 在pretrain normal的时候可以通过--normal_vgg来选择是否对normal施加vgg loss。默认只含有l1loss以及点乘loss。
3. 在pretrain albedo的时候可以通过--normal_tune来选择是否对normal进行fine tune（即改变其网络参数），默认不进行fine tune。指定--normal_tune之后，也可以指定--normal_vgg来选择是否对fine tune的normal施加vgg loss。
4. 在train整个pipeline时，可以用--normal_tune或--albedo_tune来选择是否对它们进行fine tune。同样地，可以指定--normal_vgg选择是否用这个loss对normal进行fine tune。注意，在整个pipeline中，--vgg或者--vgg_api只对shad施加loss，而albedo默认就有vgg loss和gan loss。
5. 在pretrain_normal或者非pretrain时，如果不对前面stage的net进行tune，则默认不在ckpt中保存没有更新过的前面stage的网络参数以节省空间，因此当continue时，要从前面stage的results dir去读取不会改变的网络参数。但是如果进行了tune，就会在当前stage的ckpt中保存tune完后的网络参数。

## code

### normal
``` shell
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 32770 train.py --config ./config/train.txt --batch_size 8 --pretrain_normal --train_ldr --debug_size 512 --epoch 100 --out_dir ./results_normal
```
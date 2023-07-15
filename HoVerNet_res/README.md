# HoVerNet Results

230710

## BRACS
- pre-trained model with pannuke dataset
- 5 class

```
method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : 6,
                'mode'       : 'fast',
            },
            'model_path' : '/workspace/datasource/HoVerNet_pretrained/class_pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar',
        },
        'type_info_path'  : '/workspace/github/hover_net/type_info.json',
    }

# parameter
batch_size=16
nr_gpus = torch.cuda.device_count()
nr_inference_workers=1
nr_post_proc_workers=1
mem_usage=0.99

run_args = {
        'batch_size' : int(batch_size) * nr_gpus,
        'nr_inference_workers' : int(nr_inference_workers),
        'nr_post_proc_workers' : int(nr_post_proc_workers),
        'patch_input_shape' : 256,
        'patch_output_shape' : 164,
    }

run_args.update({
        'input_dir'      : '/workspace/datasource/BRACS/previous_versions/Version1_MedIA/Images/val/6_IC/',
        'output_dir'     : '/workspace/Cell_Segment/230707_BRACS/230707_hovernet_binary/results/val_hover/6_IC/',

        'mem_usage'   : float(mem_usage),
        'draw_dot'    : True,
        'save_qupath' : False,
        'save_raw_map': True,
    })
```
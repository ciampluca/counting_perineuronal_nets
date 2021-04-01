# counting_perineuronal_nets

Experimental evaluation is based on Hydra.

In the conf folder there are two folders, corresponding to the two considered approaches (density and detection-based approaches).

In each approach folder there is a folder technique, containing base configuration yaml files for the considered techniques.
There is also a config yaml file specifying the default technique that will be used.
Finally, in the experiment folder, there are yaml configuration files corresponding to the experiment
we want to done. The values inside these files override the configuration yaml file in the technique folder.

Please note that for each experiment a folder in the outputs folder will be created. Notably, inside 
there will be also a .hydra hidden folder containing log files etc
If you use the multirun option, Hydra will create a multirun folder instad of the outputs one, having 
a similar structure.


###Examples:


- Run density based approach, using default technique (CSRNet) and configuration experiment input_size_640_640_overlap_120

        python train_density-based.py +experiment=csrnet_input_size_640_640_overlap_120

- Run two different experiments for density based approach, using default technique (CSRNet)

        python train_density-based.py +experiment=csrnet_input_size_640_640_overlap_120,csrnet_input_size_640_640_overlap_0

- Run all the experiment for density based approach using default technique (CSRNet)
        
        python train_density-based.py --multirun '+experiment=glob(csrnet*)'
        
- Run all the experiment having input size 640x640 for density based approach overriding default technique and using UNet

        python train_density-based.py technique=density_based_unet --multirun '+experiment=glob(unet*640*)'
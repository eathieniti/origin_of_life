# Complex Systems Simulation project
(MSc Computational Science-UVA)


Phospholipids in aqueous solutions arrange themselves in micelles. This is a response to their amphipathic nature; they contain both hydrophilic regions and hydrophobic.


# Discrete lattice Model

## Discrete Orientation
Run "python micelle_lattice_4orientation.py"
To obtain a simulation of the discrete model with 4 discrete orientations.

## Continuous Orientation and Micelle Migration
Run "python discrete_moving_micelle.py"
To obtain a simulation of the discrete model with continuous orientation and micelle migration.



# Continuous model 
The model with continuous motion for the lipids, but discrete timestep

## 2-point lipids 
Run "python continuous_model.py"
To obtain a simulation of the 2 point-lipids continuous model

## 3-point lipids
Run "continuous_model_3point.py"
To obtain a simulation of the 3 point-lipids continuous model

      ### Experiments
      "continuous_model_3point_min_tail_change.py"
  
        This is an extension of the 3-point lipids model to change the min_tail_distance in the same run. This is to show the transition from micelles to bilayers
  
       TODO: merge experiments code into main model or inherit functions from it..
  




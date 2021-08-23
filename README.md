# Magnet-Actuation
This repository comprises the animations as well as the final iterations of the scripts used for my Master's Thesis Individual Project.

Within the repository there will be three items.
One Python script labelled "Magnetic Actuation Animation Script.py"
One Python script labelled "Magnetic Actuation Statistics Script.py"
One folder named "Animations"

In the "Magnetic Actuation Animation Script.py" script, you will find:
- A detailed set of functions culmitating in a script capable of magnet actuation given a path
- Several prexisting paths
- Some paths are used to show basic movements with magnet actuation
- Others are used for animation purposes
- All paths are clearly seperated by which category they fall into
- To chose a path simply uncomment the chosen path and ensure that other paths have been commented out
- In addition to plotting the magnet positions and beam deformations the script will save the entire set of movements in a .gif file
- The user has the choice between seeing the physical magnets move, or seeing their relative positions shown in the animation
- Note that, just like the paths, the choice of animation is done by uncommenting the one you wish to have, and commenting the one you do not want to have
- An additional note: All symmetric paths are recommended to be recorded in 15 to 30 frames per second (fps) with the "Complex Path" being recommended to be recorded between 5 to 10 fps

In the "Magnetic Actuation Statistics Script.py" script, you will find:
- A detailed set of functions culmitating in a script capable of magnet actuation given a path
- Several collections of parameters sets to choose from
- The script will create errorbars and bar charts regarding the chosen set of parameters 
- The choice is made by uncommenting the chosen parameter set and commenting out those that were not chosen
- Note that depending on the number of parameters in the chosen set you will need to comment / uncomment the correct legend for the errorbar and distance bar charts.
- Also ensure that the correct parameter numbers appear on the aformentioned errorbars and bar charts
- The function will also plot the magnet positions and beam deformations for each parameters in the set

In the "Animations" folder, you will find:
- 12 .gif files
- Each file is an animation regarding a specific movement
- 6 files includes displaying the physical magnets, 6 do not have the physical magnets
- Those that do have the "Shape" at the end of the file name
- There are six different kinds of movement
- DL_UL: From Down Left to Up Right
- FORWARD: Moving the catheter Forward
- LEFT_RIGHT: From Left to Right
- Total: "Complex path" all movements included here
- UL_DR: From Up Left to Down Right
- UP_DOWN: From Up to Down

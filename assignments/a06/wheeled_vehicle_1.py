## =============================================================================
## PROJECT CHRONO - http:##projectchrono.org
##
## Copyright (c) 2014 projectchrono.org
## All right reserved.
##
## Use of this source code is governed by a BSD-style license that can be found
## in the LICENSE file at the top level of the distribution and at
## http://projectchrono.org/license-chrono.txt.
##
## =============================================================================
## Author: Radu Serban
## =============================================================================
##
## Wheeled vehicle tutorial
##
## =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import os
import math as m
import matplotlib.pyplot as plt
import numpy as np
## -----------------------------------------------------------------------------

# Function to add falling objects
def AddMovingObstacles(system) :
    # Create contact material, of appropriate type. Use default properties
    material = None
    if (NSC_SMC == chrono.ChContactMethod_NSC) :
        matNSC = chrono.ChMaterialSurfaceNSC()
        #Change NSC material properties as desired
        material = matNSC
    elif (NSC_SMC == chrono.ChContactMethod_SMC) :
        matSMC = chrono.ChMaterialSurfaceSMC()
        # Change SMC material properties as desired
        material = matSMC
    else:
        raise("Invalid Contact Method")
    sizeX = 300
    sizeY = 300
    height = 0
    numObstacles = 10

    for i in range(numObstacles) :
        o_sizeX = 1.0 + 3.0 * chrono.ChRandom()
        o_sizeY = 0.3 + 0.2 * chrono.ChRandom()
        o_sizeZ = 0.05 + 0.1 * chrono.ChRandom()
        obstacle = chrono.ChBodyEasyBox(o_sizeX, o_sizeY, o_sizeZ, 2000.0, True, True, material)

        o_posX = (chrono.ChRandom() - 0.5) * 0.4 * sizeX
        o_posY = (chrono.ChRandom() - 0.5) * 0.4 * sizeY
        o_posZ = height + 4
        rot = chrono.ChQuaternionD(chrono.ChRandom(), chrono.ChRandom(), chrono.ChRandom(), chrono.ChRandom())
        rot.Normalize()
        obstacle.SetPos(chrono.ChVectorD(o_posX, o_posY, o_posZ))
        obstacle.SetRot(rot)

        system.AddBody(obstacle)
    
## -----------------------------------------------------------------------------

# Function to add obstacles
def AddFixedObstacles(system) :
    # Create contact material, of appropriate type. Use default properties
    material = None
    if (NSC_SMC == chrono.ChContactMethod_NSC) :
        matNSC = chrono.ChMaterialSurfaceNSC()
        #Change NSC material properties as desired
        material = matNSC
    elif (NSC_SMC == chrono.ChContactMethod_SMC) :
        matSMC = chrono.ChMaterialSurfaceSMC()
        # Change SMC material properties as desired
        material = matSMC
    else:
        raise("Invalid Contact Method")

    for i in range(8) :
        obstacle = chrono.ChBodyEasyCylinder(0.5, 6.0, 2000, True, True, material)
        obstacle.SetPos(chrono.ChVectorD(-2.0 * i + 20, 0, -0.4))
        
        obstacle.SetRot(chrono.Q_from_AngAxis(15 * chrono.CH_C_DEG_TO_RAD, chrono.VECT_Y))
        obstacle.SetBodyFixed(True);
        system.AddBody(obstacle);
        
## =============================================================================

# JSON-specification files for vehicle system and subsystems

#### ---------------------------------------------------------------------------
#### EXERCISE 2
#### Write a vehicle JSON specification file named 'WheeledVehicle_mod.json'
#### which uses a reduced double wishbone suspension in the front and a solid
#### axle suspension in the rear.
#### - complete the skeleton of 'DoubleWishbone_reduced.json'
#### - copy the existing WheeledVehicle.json to WheeledVehicle_mod.json
#### - edit WheeledVehicle_mod.json to use the new front and rear suspensions 
#### ---------------------------------------------------------------------------

vehicle_file = "vehicle/WheeledVehicle_mod.json"
rigidtire_file = "vehicle/RigidTire.json"
simplepowertrain_file = "vehicle/SimplePowertrain.json"
rigidterrain_file = "terrain/RigidPlane.json"

# Initial vehicle position and orientation
initLoc = chrono.ChVectorD(0, 0, 1.0)
initRot = chrono.ChQuaternionD(1, 0, 0, 0)

# Simulation step size
step_size = 1e-3

# Time interval between two render frames
render_step_size = 1.0 / 50    # FPS = 50

# Point on chassis tracked by the camera
trackPoint = chrono.ChVectorD(0.0, 0.0, 1.75)

# Contact method
NSC_SMC = chrono.ChContactMethod_SMC


# ----------------------
# Set path to data files
# ----------------------

# Path to Chrono data files (textures, fonts, etc.)
chrono.SetChronoDataPath("./data/")

# Path to the data files for this vehicle (JSON specification files)
veh.SetDataPath("./data/")

# --------------------------
# Create the various modules
# --------------------------
    
# Create and initialize the vehicle system
vehicle = veh.WheeledVehicle(veh.GetDataFile(vehicle_file), NSC_SMC)

vehicle.Initialize(chrono.ChCoordsysD(initLoc, initRot))

vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
vehicle.SetWheelVisualizationType(veh.VisualizationType_NONE)

# Create the terrain
terrain = veh.RigidTerrain(vehicle.GetSystem(), veh.GetDataFile(rigidterrain_file))
AddFixedObstacles(vehicle.GetSystem())
##AddMovingObstacles(vehicle.GetSystem())

# Create and initialize the powertrain system
powertrain = veh.SimplePowertrain(veh.GetDataFile(simplepowertrain_file))
vehicle.InitializePowertrain(powertrain)

# Create and initialize the tires
for axle in vehicle.GetAxles() :
    tireL = veh.RigidTire(veh.GetDataFile(rigidtire_file))
    tireR = veh.RigidTire(veh.GetDataFile(rigidtire_file))
    vehicle.InitializeTire(tireL, axle.m_wheels[0], veh.VisualizationType_MESH)
    vehicle.InitializeTire(tireR, axle.m_wheels[1], veh.VisualizationType_MESH)
    
# Create the Irrlicht vehicle application
app = veh.ChVehicleIrrApp(vehicle, "Vehicle Demo")

app.AddTypicalLights()
app.SetChaseCamera(trackPoint, 6.0, 0.5)

app.SetTimestep(step_size)

app.AssetBindAll()
app.AssetUpdateAll()

#### ---------------------------------------------------------------------------
#### EXERCISE 1
#### Create a driver system that reads vehicle inputs from a data file
#### - the type of this driver vehicle is ChDataDriver
#### - you construct the driver by specifying two arguments to ChDataDriver:
####   (1) the vehicle it is associated with
####   (2) the name of the input data file
#### - use the provided input data file 'DriverData.txt' in the 'data/driver/'
####   directory 
#### - Make sure you give the proper path to that file!
####   HINT: see how the JSON files for the vehicle are specified and used
####   elsewhere in this Python script.
#### ---------------------------------------------------------------------------
DriverDataPath=veh.GetDataPath()+"/driver/DriverData.txt"
driver=veh.ChDataDriver(vehicle,DriverDataPath)

# ---------------
# Simulation loop
# ---------------

# Number of simulation steps between two 3D view render frames
render_steps = m.ceil(render_step_size / step_size)

#Initialize simulation frame counter and simulation time
step_number = 0
time = 0

realtime_timer = chrono.ChRealtimeStepTimer()
x_pos=[]
y_pos=[]
z_pos=[]
sim_time=[]
while (app.GetDevice().run())  :
    # Render scene
    if (step_number % render_steps == 0) :
        app.BeginScene(True, True, chronoirr.SColor(255, 140, 161, 192))
        app.DrawAll();
        app.EndScene();
        
    # Get driver inputs
    driver_inputs = driver.GetInputs()

    # Update modules (process inputs from other modules)
    time = vehicle.GetSystem().GetChTime()
    driver.Synchronize(time)
    vehicle.Synchronize(time, driver_inputs, terrain)
    terrain.Synchronize(time)
    app.Synchronize("", driver_inputs)

    # Advance simulation for one timestep for all modules
    driver.Advance(step_size)
    vehicle.Advance(step_size)
    terrain.Advance(step_size)
    app.Advance(step_size)

    if time<10:
        current_position=vehicle.GetVehiclePos()
        x_pos.append(current_position.x)
        y_pos.append(current_position.y)
        z_pos.append(current_position.z)
        sim_time.append(time)

    # Increment step number
    step_number += 1
    # Spin in place for real time to catch up
    realtime_timer.Spin(step_size)

    if (time > 12):
        break
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x_pos,y_pos)
ax1.set(ylabel='y-position [m]',xlabel='x-position [m]')
ax1.set_title("Trajectory")
ax1.grid()
ax2.plot(sim_time,z_pos)
ax2.set(ylabel='height [m]',xlabel='time [s]')
ax2.set_title("Height")
ax2.grid()
plt.tight_layout()
plt.show()

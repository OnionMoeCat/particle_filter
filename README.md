# Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data.

By implementing the particle filter, the robot will locate itself, which is expected to be reasonably accurate.

# Running the Code

Run ./install_%your_platform%.sh to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built by ./build.sh

After it is built, run by ./run.sh

Open the simulator, start the kidnapped vehicle simulator, press "start" button.

You can see the car starts moving, and the nearby landmark are linked with the car. The error is about:

x: 0.1, y: 0.1, theta: 0.03

The simulator passed the test after about 100 seconds:
```
Success! Your particle filter passed!
```

![alt text](images/result.png)


# Implementation of the particle filter

The only file I modify is `particle_filter.cpp` in the `src` directory. I am going to cover all the works needed below: 

## Initialization

## Prediction
This is done in:
```
  void prediction(double delta_t, double std_pos[], double velocity, 
                  double yaw_rate);
```

Prediction Predicts the state for the next time step using the process model.
The formula of model is:
```
      particle.x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta = particle.theta + yaw_rate * delta_t;
```
where the particle's state is calculated using its previous state, angular velocity (yaw_rate), and delta time.
If angular_velocity(yaw_rate) is close to 0, we are facing a divide by zero problem, this is handled by
```
    if (fabs(yaw_rate) < 0.000001) 
    {
      particle.x = particle.x + velocity * delta_t * cos(particle.theta);
      particle.y = particle.y + velocity * delta_t * sin(particle.theta);
    }
```
The noise is applied afterwards, which is a gaussian noise.

## Update weights

This is done in
```
  void updateWeights(double sensor_range, double std_landmark[], 
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);
```

For each particle, updating weight of the partile needs these steps:
1. Make prediction for each landmarks within sensor range
2. transform the landmark from vehicle coordinate system to map relative systemm
3. Associate the observed landmark with predictions
4. Compute weights using associated observations

### Make prediction for each landmarks within sensor range
Iterate through all the landmarks in the map, add the landmark to predictions if the distance from the particle is within the sensor range.
As shown below:
```
	      double dis_sq = (landmark.x_f - particle.x) * (landmark.x_f - particle.x)
		+ (landmark.y_f - particle.y) * (landmark.y_f - particle.y);
	      if (dis_sq < sensor_range * sensor_range)
          { // the landmark is within the sensor range, add to the prediction vector
```

### Transform the landmark from vehicle coordinate system to map relative systemm
We need to make sure both predictions and observations are in the map relative system. The input observation is in vehicle coordinate system,
and the transform is:
```
	      observations_in_map[i].x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
	      observations_in_map[i].y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
```
### Associate the observed landmark with predictions
Then we associate observation with predictions
We call the function   
```
void dataAssociation(std::vector<LandmarkObs> predicted, 
                     std::vector<LandmarkObs>& observations);
```
Each observation is associated by the closest prediction landmark, this is done by:
```
        observation.id = prediction.id;
```
### Compute weights using associated observations
Afterwards, we calculate the weight of this particle using associated observations
The weight of the particle is the multiply of weight of each observation.
For each observation, the weight is evaluated by how close the observation is from the prediction.
Specifically, the weight is the probablity of the multivariable gaussian distribution with prediction as the average.
```
multiv_prob(std_x, std_y, obs_x, obs_y, mu_x, mu_y);
```
mu_x, mu_y is the x,y of the prediction, obs_x,obs_y is the x,y of observation. std_x, std_y is the standard deviation of the distribution.

Basically, the weight of the particle is higher if the observation is closer to the prediction.

## Resample
We resample the group of particles by making higher weight particles appear more often in the resampled set.
This is done by 
```
  // resample the particles according to its weight
  std::default_random_engine gen;
  std::discrete_distribution<size_t> dist_index(weights.begin(), weights.end());
  vector<Particle> resampled_particles(particles.size());
  for (auto i = 0; i < particles.size(); i ++)
  {
      resampled_particles[i] = particles[dist_index(gen)];
  }
  particles = resampled_particles;
```
We have a discrete distribution taking all particles' weight, and index of each resampled particles is generated by this discrete distribution.
Thus, particle's frequency in the resampled set is proportional to its weight.
By doing this, we get a better group of particles.
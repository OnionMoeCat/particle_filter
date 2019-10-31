/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. From experiment, 1024 gives
  // the best performance
  num_particles = 1024;
  
  std::default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  // Setup the normal distribution for x, y, theta
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);
  
  // Setup each particle, whose x, y, theta is generated from the gaussian distribution
  particles.resize(num_particles);
  for (int i = 0; i < num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    auto & particle = particles[i];
    particle.id = i;
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
  }
  
  weights.resize(num_particles);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  std::default_random_engine gen;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  // Setup the normal distribution for x, y, theta
  std::normal_distribution<double> dist_x(0, std_x);
  std::normal_distribution<double> dist_y(0, std_y);
  std::normal_distribution<double> dist_theta(0, std_theta);
  
  for (int i = 0; i < num_particles; ++i)
  {
   auto & particle = particles[i];
    if (fabs(yaw_rate) < 0.000001) 
    {
      particle.x = particle.x + velocity * delta_t * cos(particle.theta);
      particle.y = particle.y + velocity * delta_t * sin(particle.theta);
    }
    else
    {
      particle.x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    }
    
    particle.theta = particle.theta + yaw_rate * delta_t;
    
    double x_noise = dist_x(gen);
    double y_noise = dist_y(gen);
    double theta_noise = dist_theta(gen);
    
    particle.x += x_noise;
    particle.y += y_noise;
    particle.theta += theta_noise;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // For each observation, find the closest prediction, and associate with that landmark
  for (auto & observation : observations)
  {
    double min_dist = std::numeric_limits<double>::max();
    observation.id = -1;
    for (auto i = 0; i < predicted.size(); ++i)
    {
      const auto & prediction = predicted[i];
      double cur_dist = (prediction.x - observation.x) * (prediction.x - observation.x)
        + (prediction.y - observation.y) * (prediction.y - observation.y);
      if (cur_dist < min_dist)
      {
        min_dist = cur_dist;
        observation.id = prediction.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

   double std_x = std_landmark[0];
   double std_y = std_landmark[1];
   // update the weight for each particle.
   for (int j = 0; j < particles.size(); ++ j)
   {
        auto & particle = particles[j];
        // make prediction for each landmarks
	    vector<LandmarkObs> predictions_in_map;
        for (const auto & landmark : map_landmarks.landmark_list)
        {
	      // add to the prediction is the landmark is in the sensor range
	      double dis_sq = (landmark.x_f - particle.x) * (landmark.x_f - particle.x)
		+ (landmark.y_f - particle.y) * (landmark.y_f - particle.y);
	      if (dis_sq < sensor_range * sensor_range)
	      {
	        LandmarkObs prediction_landmark = {
			.id = landmark.id_i,
			.x = static_cast<double>(landmark.x_f),
			.y = static_cast<double>(landmark.y_f)
		    };
		    predictions_in_map.push_back(prediction_landmark);
          }
        }

	vector<LandmarkObs> observations_in_map(observations.size());
	for (auto i = 0; i < observations.size(); ++ i)
	{
          const auto & observation = observations[i];
          // transform the landmark from vehicle coordinate system to map relative system
	      observations_in_map[i].x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
	      observations_in_map[i].y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
	}	

    // from landmark observations, associate them with the predictions in the map
    dataAssociation(predictions_in_map, observations_in_map);
	
	vector<int> associations;
	vector<double> sense_x;
	vector<double> sense_y;
	
	// compute weights using the associated observations
	double weight = 1.0;
	for (const auto & observation : observations_in_map)
	{
	      double obs_x = observation.x;
	      double obs_y = observation.y;
	      int index = observation.id;

	      associations.push_back(index);
	      sense_x.push_back(obs_x);
	      sense_y.push_back(obs_y);

	      double mu_x = static_cast<double>(map_landmarks.landmark_list[index - 1].x_f);
	      double mu_y = static_cast<double>(map_landmarks.landmark_list[index - 1].y_f);     
	      weight *= multiv_prob(std_x, std_y, obs_x, obs_y, mu_x, mu_y);
	}
	particle.weight = weight;
    weights[j] = weight;
     
    // store the association information in the particle
	SetAssociations(particle, associations, sense_x, sense_y);
   }
}

void ParticleFilter::resample() {
  // resample the particles according to its weight
  std::default_random_engine gen;
  std::discrete_distribution<size_t> dist_index(weights.begin(), weights.end());
  vector<Particle> resampled_particles(particles.size());
  for (auto i = 0; i < particles.size(); i ++)
  {
      resampled_particles[i] = particles[dist_index(gen)];
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
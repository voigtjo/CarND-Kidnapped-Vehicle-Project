/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 250;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	default_random_engine gen;

	normal_distribution<double> N_x(x, std_x);
	normal_distribution<double> N_y(y, std_y);
	normal_distribution<double> N_theta(theta, std_theta);

	particles.reserve(num_particles);
	weights.reserve(num_particles);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (auto& particle : particles)
	{

		double new_x;
		double new_y;
		double new_theta;

		if (fabs(yaw_rate) < 0.0001)
		{
			new_x = particle.x + velocity*delta_t*cos(particle.theta);
			new_y = particle.y + velocity*delta_t*sin(particle.theta);
			new_theta = particle.theta;
		}
		else
		{
			new_x = particle.x + velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
			new_y = particle.y + velocity/yaw_rate*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
			new_theta = particle.theta + yaw_rate*delta_t;
		}

		normal_distribution<double> N_x(new_x, std_pos[0]);
		normal_distribution<double> N_y(new_y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);

		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	int i = 0;
	for (Particle particle : particles){
		particle.weight = 1.0;

		vector<LandmarkObs> converted_Observations;
		converted_Observations.reserve(observations.size());
		for (LandmarkObs currentObs : observations){
			LandmarkObs convertedObs;

			//Convert vehicle to map coordinates
			convertedObs.x = particle.x+(currentObs.x*cos(particle.theta)-currentObs.y*sin(particle.theta));
			convertedObs.y = particle.y+(currentObs.x*sin(particle.theta)+currentObs.y*cos(particle.theta));
			converted_Observations.push_back(convertedObs);
		}

		for (const auto& transObs : converted_Observations){
			double closet_dis = sensor_range;
			int association = 0;


			float x_f = NAN;
			float y_f = NAN;

			for (const auto& current_Landmark : map_landmarks.landmark_list){
				double calc_dist = sqrt(pow(transObs.x-current_Landmark.x_f,2.0)+pow(transObs.y-current_Landmark.y_f,2.0));
				if(calc_dist < closet_dis){
					closet_dis = calc_dist;
					x_f = current_Landmark.x_f;
					y_f = current_Landmark.y_f;
				}
			}

			if(!isnan(x_f) &&  !isnan(y_f)){
				double meas_x = transObs.x;
				double meas_y = transObs.y;
				double mu_x = x_f;
				double mu_y = y_f;
				long double multipler = 1/(2*M_PI*std_landmark[0]*std_landmark[1])
            		*exp(-(pow(meas_x-mu_x,2.0)/(2*pow(std_landmark[0],2.0))+pow(meas_y-mu_y,2.0)/(2*pow(std_landmark[1],2.0))));
				if(multipler > 0){
					particle.weight*= multipler;
				}

			}
		}
		weights[i] = particle.weight;
		i++;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resampled_particles;
	resampled_particles.reserve(particles.size());
	for (int i = 0; i < particles.size(); i++){
		resampled_particles.push_back(particles[distribution(gen)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}

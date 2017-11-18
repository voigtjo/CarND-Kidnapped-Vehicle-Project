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
	// observed measurement to this particular landmark.
	for (auto& obs : observations)
    {
        // init
        double min_dist = numeric_limits<double>::max();
        int map_id = -1;

        for (auto& pred : predicted)
        {
            // Calc distance between current and predicted landmarks
            double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
            // Find the predicted landmark nearest the current observed landmark
            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                map_id = pred.id;
            }
        }
        // Assign the observed measurement to this particular landmark
        obs.id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution.
	for (auto& particle : particles)
    {
      // Get the particle x, y coordinates
      double px = particle.x;
      double py = particle.y;
      double ptheta = particle.theta;
      vector<LandmarkObs> predictions;

      // For each map landmark

      for (auto& landmark: map_landmarks.landmark_list)
      {
        // Get id and x,y coordinates
        float lx = landmark.x_f;
        float ly = landmark.y_f;

        // Consider landmarks within sensor range of the particle
        float x_diff = fabs(lx - px);
        float y_diff = fabs(ly - py);
        if (x_diff <= sensor_range && y_diff <= sensor_range)
        {
          // Add prediction
          predictions.push_back(LandmarkObs{landmark.id_i, lx, ly });
        }
      }

      // List of transformed observations from vehicle coordinates to map coordinates
      vector<LandmarkObs> transformed_obs;
      for (auto& obs : observations)
      {
        double tx = cos(ptheta)*obs.x - sin(ptheta)*obs.y + px;
        double ty = sin(ptheta)*obs.x + cos(ptheta)*obs.y + py;
        transformed_obs.push_back(LandmarkObs{obs.id, tx, ty });
      }

      // Data association for the predictions and transformed observations
      dataAssociation(predictions, transformed_obs);
      // Re-init weight
      particle.weight = 1.0;
      for (auto& trans_obs : transformed_obs)
      {
        double x, y;
        double mu_x = trans_obs.x;
        double mu_y = trans_obs.y;
        // Get x,y coordinates of the prediction associated with the current observation
        for (auto& pred : predictions)
        {
          if (pred.id == trans_obs.id)
          {
            x = pred.x;
            y = pred.y;
          }
        }
        // Calculate weight for this observation with multivariate Gaussian
        double sigx = std_landmark[0];
        double sigy = std_landmark[1];
        double gauss_norm =  1/(2 * M_PI * sigx * sigy);
        double exponent  = (pow(x-mu_x, 2) / (2 * sigx * sigx)) + (pow(y-mu_y, 2) / (2 * sigy * sigy));
        double obs_weight = gauss_norm * exp(-exponent);
        // Product of this obersvation weight with total observations weight
        particle.weight *= obs_weight;
      }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
    vector<Particle> new_particles;
    default_random_engine gen;
    vector<double> weights;
    for (auto& particle : particles)
    {
        weights.push_back(particle.weight);
    }
    // Get max weight
    double max_weight = *max_element(weights.begin(), weights.end());
    // Generate random starting index for resampling wheel
    uniform_int_distribution<int> uni_dist(0, num_particles-1);
    uniform_real_distribution<double> uni_real_dist(0.0, max_weight);
    int index = uni_dist(gen);
    double beta = 0.0;
    // Resampling wheel
    for (auto& particle : particles)
    {
        beta += uni_real_dist(gen) * 2.0;
        while (weights[index] < beta)
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
     }
     particles = new_particles;
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

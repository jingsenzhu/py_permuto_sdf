#pragma once


#include "torch/torch.h"


class Sphere{
public:
    Sphere(const float radius, const torch::Tensor &center);
    ~Sphere();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  ray_intersection(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs);
    torch::Tensor rand_points_inside(const int nr_points);
    torch::Tensor check_point_inside_primitive(const torch::Tensor& points);


    float m_radius;
    torch::Tensor m_center_tensor;

private:
    

  
};

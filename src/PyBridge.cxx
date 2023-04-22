// #include "permuto_sdf/PyBridge.h"

#include "torch/torch.h"
#include <torch/extension.h>


//my stuff 
#include "permuto_sdf/Sphere.cuh"
#include "permuto_sdf/OccupancyGrid.cuh"
#include "permuto_sdf/VolumeRendering.cuh"
#include "permuto_sdf/RaySampler.cuh"
#include "permuto_sdf/RaySamplesPacked.cuh"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(py_permuto_sdf, m) {

    py::class_<Sphere> (m, "Sphere")
    .def(py::init<const float, const torch::Tensor&>())
    .def("ray_intersection", &Sphere::ray_intersection ) 
    .def("rand_points_inside", &Sphere::rand_points_inside, py::arg("nr_points") ) 
    .def("check_point_inside_primitive", &Sphere::check_point_inside_primitive ) 
    .def_readwrite("m_center_tensor", &Sphere::m_center_tensor ) 
    .def_readwrite("m_radius", &Sphere::m_radius ) 
    ;


    py::class_<OccupancyGrid> (m, "OccupancyGrid")
    .def(py::init<const int, const float, const torch::Tensor&>())
    .def_static("make_grid_values", &OccupancyGrid::make_grid_values ) 
    .def_static("make_grid_occupancy", &OccupancyGrid::make_grid_occupancy ) 
    .def("set_grid_values", &OccupancyGrid::set_grid_values ) 
    .def("set_grid_occupancy", &OccupancyGrid::set_grid_occupancy ) 
    .def("get_grid_values", &OccupancyGrid::get_grid_values ) 
    .def("get_grid_occupancy", &OccupancyGrid::get_grid_occupancy ) 
    .def("get_nr_voxels", &OccupancyGrid::get_nr_voxels ) 
    .def("get_nr_voxels_per_dim", &OccupancyGrid::get_nr_voxels_per_dim ) 
    .def("compute_grid_points", &OccupancyGrid::compute_grid_points ) 
    .def("compute_random_sample_of_grid_points", &OccupancyGrid::compute_random_sample_of_grid_points ) 
    .def("check_occupancy", &OccupancyGrid::check_occupancy ) 
    .def("update_with_density", &OccupancyGrid::update_with_density ) 
    .def("update_with_density_random_sample", &OccupancyGrid::update_with_density_random_sample ) 
    .def("update_with_sdf", &OccupancyGrid::update_with_sdf ) 
    .def("update_with_sdf_random_sample", &OccupancyGrid::update_with_sdf_random_sample ) 
    .def("update_with_sdf_positions", &OccupancyGrid::update_with_sdf_positions ) 
    .def("compute_samples_in_occupied_regions", &OccupancyGrid::compute_samples_in_occupied_regions ) 
    .def("compute_first_sample_start_of_occupied_regions", &OccupancyGrid::compute_first_sample_start_of_occupied_regions ) 
    .def("advance_sample_to_next_occupied_voxel", &OccupancyGrid::advance_sample_to_next_occupied_voxel ) 
    ;

    py::class_<RaySamplesPacked> (m, "RaySamplesPacked")
    .def(py::init<const int, const int>())
    .def("compact_to_valid_samples", &RaySamplesPacked::compact_to_valid_samples ) 
    .def("compute_exact_nr_samples", &RaySamplesPacked::compute_exact_nr_samples ) 
    .def("initialize_with_one_sample_per_ray", &RaySamplesPacked::initialize_with_one_sample_per_ray ) 
    .def("set_sdf", &RaySamplesPacked::set_sdf ) 
    .def("remove_sdf", &RaySamplesPacked::remove_sdf ) 
    .def_readwrite("samples_pos",  &RaySamplesPacked::samples_pos )
    .def_readwrite("samples_pos_4d",  &RaySamplesPacked::samples_pos_4d )
    .def_readwrite("samples_dirs",  &RaySamplesPacked::samples_dirs )
    .def_readwrite("samples_z",  &RaySamplesPacked::samples_z )
    .def_readwrite("samples_dt",  &RaySamplesPacked::samples_dt )
    .def_readwrite("samples_sdf",  &RaySamplesPacked::samples_sdf )
    .def_readwrite("ray_start_end_idx",  &RaySamplesPacked::ray_start_end_idx )
    .def_readwrite("ray_fixed_dt",  &RaySamplesPacked::ray_fixed_dt )
    .def_readwrite("max_nr_samples",  &RaySamplesPacked::max_nr_samples )
    .def_readwrite("cur_nr_samples",  &RaySamplesPacked::cur_nr_samples )
    .def_readwrite("rays_have_equal_nr_of_samples",  &RaySamplesPacked::rays_have_equal_nr_of_samples )
    .def_readwrite("fixed_nr_of_samples_per_ray",  &RaySamplesPacked::fixed_nr_of_samples_per_ray )
    ;

    py::class_<VolumeRendering> (m, "VolumeRendering")
    .def(py::init<>())
    .def_static("volume_render_nerf", &VolumeRendering::volume_render_nerf ) 
    .def_static("compute_dt", &VolumeRendering::compute_dt ) 
    .def_static("cumprod_alpha2transmittance", &VolumeRendering::cumprod_alpha2transmittance ) 
    .def_static("integrate_with_weights", &VolumeRendering::integrate_with_weights ) 
    .def_static("sdf2alpha", &VolumeRendering::sdf2alpha ) 
    .def_static("sum_over_each_ray", &VolumeRendering::sum_over_each_ray ) 
    .def_static("cumsum_over_each_ray", &VolumeRendering::cumsum_over_each_ray ) 
    .def_static("compute_cdf", &VolumeRendering::compute_cdf )  
    .def_static("importance_sample", &VolumeRendering::importance_sample )  
    .def_static("combine_uniform_samples_with_imp", &VolumeRendering::combine_uniform_samples_with_imp )  
    //backward passes
    .def_static("volume_render_nerf_backward", &VolumeRendering::volume_render_nerf_backward ) 
    .def_static("cumprod_alpha2transmittance_backward", &VolumeRendering::cumprod_alpha2transmittance_backward )  
    .def_static("integrate_with_weights_backward", &VolumeRendering::integrate_with_weights_backward )  
    .def_static("sum_over_each_ray_backward", &VolumeRendering::sum_over_each_ray_backward )  
    ;

    py::class_<RaySampler> (m, "RaySampler")
    .def(py::init<>())
    .def_static("compute_samples_fg", &RaySampler::compute_samples_fg ) 
    .def_static("compute_samples_bg", &RaySampler::compute_samples_bg ) 
    ;

}




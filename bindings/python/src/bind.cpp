#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vault.hpp"

namespace py = pybind11;

// Zero-copy NumPy to OpenCV Mat conversion
cv::Mat numpy_uint8_to_cv_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    // Assuming 2D image (Grayscale) or 3D (BGR)
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int type = (buf.ndim == 3) ? CV_8UC3 : CV_8UC1;
    return cv::Mat(rows, cols, type, (unsigned char*)buf.ptr);
}

PYBIND11_MODULE(_core, m) {
    py::class_<OrbConfig>(m, "OrbConfig")
        .def(py::init<>())
        .def_readwrite("max_features", &OrbConfig::max_features)
        .def_readwrite("fast_cells", &OrbConfig::fast_cells)
        .def_readwrite("deep_cells", &OrbConfig::deep_cells)
        .def_readwrite("max_hamming_distance", &OrbConfig::max_hamming_distance);

    py::class_<MatchResult>(m, "MatchResult")
        .def_readonly("label", &MatchResult::label)
        .def_readonly("confidence", &MatchResult::confidence)
        .def_readonly("fallback_used", &MatchResult::fallback_used);

    py::class_<Vault>(m, "Vault")
        .def(py::init<const OrbConfig&>())
        .def("add", [](Vault& self, py::array_t<uint8_t> img, const std::string& label) {
            self.add(numpy_uint8_to_cv_mat(img), label);
        })
        // Release the GIL during training so Python doesn't freeze
        .def("build", &Vault::build, py::call_guard<py::gil_scoped_release>())
        .def("search", [](Vault& self, py::array_t<uint8_t> img) {
            return self.search(numpy_uint8_to_cv_mat(img));
        });
}

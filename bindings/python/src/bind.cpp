#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vault.hpp"

namespace py = pybind11;

// Zero-copy NumPy to OpenCV Mat conversion
cv::Mat numpy_uint8_to_cv_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    if (buf.ndim < 2 || buf.ndim > 3)
        throw std::invalid_argument("Image array must be 2D (grayscale) or 3D (BGR).");
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
        .def_readwrite("max_hamming_distance", &OrbConfig::max_hamming_distance)
        .def_readwrite("confidence_threshold", &OrbConfig::confidence_threshold)
        .def_readwrite("resize_dim", &OrbConfig::resize_dim);

    py::class_<MatchResult>(m, "MatchResult")
        .def_readonly("label", &MatchResult::label)
        .def_readonly("confidence", &MatchResult::confidence)
        .def_readonly("fallback_used", &MatchResult::fallback_used);

    py::class_<Vault>(m, "Vault")
        .def(py::init<const OrbConfig&>())
        .def("add", [](Vault& self, py::array_t<uint8_t> img, const std::string& label) {
            self.add(numpy_uint8_to_cv_mat(img), label);
        })
        .def("add_batch", [](Vault& self, py::list images, py::list labels) {
            std::vector<cv::Mat> mats;
            std::vector<std::string> strs;
            mats.reserve(images.size());
            strs.reserve(labels.size());
            for (auto img : images) {
                auto arr = py::cast<py::array_t<uint8_t>>(img);
                mats.push_back(numpy_uint8_to_cv_mat(arr));
            }
            for (auto label : labels)
                strs.push_back(py::cast<std::string>(label));
            {
                py::gil_scoped_release release;
                self.add_batch(mats, strs);
            }
        })
        // Release the GIL during training so Python doesn't freeze
        .def("build", &Vault::build, py::call_guard<py::gil_scoped_release>())
        .def("search", [](Vault& self, py::array_t<uint8_t> img) {
            return self.search(numpy_uint8_to_cv_mat(img));
        })
        .def("save", &Vault::save)
        .def("load", &Vault::load);
}

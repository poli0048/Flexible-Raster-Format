# FRF - The Flexible Raster Format
FRF is a file format for raster imagery that supports an arbitrary number of layers, arbitrary bit-depth integer data along with 32-bit and 64-bit floating-point data, and unambiguous geo-registration and geo-tagging. Key features include:

 - Support for unpadded, arbitrary bit-dept unsigned integers (up to 64 bits), which makes it ideal for saving raw image data from imagers with ADCs above 8-bits without wasting unreasonable amounts of space. It also has optimized accessors for data types that use integer numbers of bytes, for situations where performance is a greater concern than storage efficiency.
 - Support for Geo-registration (associating each pixel with a specific location on Earth) as well as Geo-tagging (recording the position, time, and/or orientation of a camera when an image is taken). Both types of metadata are supported in a standardized form and the complexity of registration is handled internally - the library lets you simply query the WGS84 GPS coordinates of a pixel without worrying about the details of how registration data is stored. If you are looking for an alternative to GeoTiff, you should consider FRF.
 - FRF is designed to represent data in arbitrary numeric ranges, regardless of the underlying storage type. We make a distinction between the "raw value" of a pixel (for instance, an integer 0-255) and the "value" of a pixel, which is computed as an affine function of the raw value using coefficients saved in the image metadata. This supports the representation of meaningful data, optionally with SI units attached.
 
# Current State
FRF is developed and maintained by Bryan Poling at Sentek Systems, LLC. FRF is not a finished spec, although it is currently used by the Cheetah Structure from Motion software package (Sentek Systems, LLC) for representing stitched aerial imagery from multi-spectral cameras.

# Using FRF
The FRF library consists of a single C++ header file and a single (large) C++ source file. Just include the source file in your project and you are good to go. The only external dependency that FRF has is Eigen. FRF has been tested with GCC under Linux and with MSVC under Windows.

# License
FRF is available under a permissive 3-clause BSD license.

# More Info
See "FRF Draft Spec.pdf" in this repository for more detailed information on the specification. See "Examples" to get started using FRF.

# Custom Block Codes
FRF supports storing custom metadata in the image header through "Custom Information Blocks". These are only meant for storing additional info that can't be encoded using standard FRF functionality. The FRF library treats a custom info block as sequence of bytes that you must create and interpret yourself, so if you use custom info blocks you may be the only one who can make sense of the data you store in them. A custom information block has a payload that begins with a 64-bit unsigned integer called the "Custom Block Code" (full details in "FRF Draft Spec.pdf"). This code is used to differentiate between different kinds of custom info blocks. These are not regulated, but we maintain a list of known custom block codes to avoid conflicts. If you are going to make your own custom blocks it is strongly recommended that you use a custom block code that isn't already in use and contact me to get your code listed in our table. Here is a table of known custom block codes being used in the wild:

| CustomBlockCode (Uint64) | Defined By | Used For (& Link) |
| ------------------------ | ---------- | ------------------------|
| 0 | Sentek Systems, LLC | Shadow Map Information Block (NIFA Grant # 20206702130758) |


//This small program creates a simple minimal FRF image and saves it. It then loads it and prints out the value at a few different pixels locations.
//Author: Bryan Poling
//Copyright (c) 2020 Sentek Systems, LLC. All rights reserved. 

//System Includes
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <bitset>
#include <chrono>
#include <random>

//Project Includes
#include "../FRF.h"

static void CreateImageAndSaveIt(std::string Filename) {
	FRFImage myImage; //Create a new FRF Image
	
	//Set Image dimensions - this must be done now, when there are no layers in the image yet.
	if (! myImage.SetWidth(2560U))
		std::cerr << "Failed to set image width.\r\n";
	if (! myImage.SetHeight(3600U))
		std::cerr << "Failed to set image height.\r\n";
	
	//Add a new layer and set it up
	FRFLayer * newLayer = myImage.AddLayer();
	newLayer->Name = std::string("My First Layer");
	newLayer->Description = std::string("This is the first layer in my first FRF image!");
	newLayer->UnitsCode = 3; //Set units to cubic meters... the value at each pixel will represent a volume
	newLayer->SetTypeCode(55U); //See Table 1 in the spec. We are going to use 55-bit unsigned integers for each pixel in this layer
	newLayer->HasValidityMask = false; //All pixels will be assumed valid
	newLayer->SetAlphaAndBetaForGivenRange(0.0, 5.0); //Let the library figure out my layer coefficients for me - I want to represent volumes between 0 and 5 m^3
	newLayer->AllocateStorage(); //This needs to be called before the layer can be accessed
	
	//Zero-fill the new layer. Notice that we are setting each pixel based on the value we care about... we let the library figure out the raw value based on alpha + beta
	for (uint32_t row = 0U; row < myImage.Rows(); row++) {
		for (uint32_t col = 0U; col < myImage.Cols(); col++)
			newLayer->SetValue(row, col, 0.0);
	}
	
	//Set a few values
	newLayer->SetValue(73U, 24U, 3.225);
	newLayer->SetValue(12U, 82U, 2.984);
	newLayer->SetValue(31U, 3900U, 3.225); //Out of bounds setting is safe, but won't do anything
	newLayer->SetValue(19U, 122U, -7.5); //A value beneath 0 or above 5 will saturate to the closest achievable value... 0 in this case.
	newLayer->SetValue(19U, 123U, 15.3); //Similarly, this will saturate to 5. If we had specified a different range for the layer we would get a different result
	
	//Add a default visualization - we always need at least 1 visualization, even if we are saving data and aren't really interested in how the imagery "looks".
	//The default visualization ensures that no matter who opens this file, when, where, or with what software... they will see the same thing you see.
	FRFVisualizationRGB * viz = myImage.AddVisualizationRGB();
	viz->Name        = std::string("My RGB Viz");
	viz->Description = std::string("A dummy RGB viz that uses layer 0 for all channels");
	viz->RedIndex    = 0U;
	viz->RedMin      = 0.0; //Value of 0 will correspond to no red
	viz->RedMax      = 5.0; //Value of 5 will correspond to max red
	viz->GreenIndex  = 0U;  //...
	viz->GreenMin    = 0.0; //...
	viz->GreenMax    = 5.0; //...
	viz->BlueIndex   = 0U;  //...
	viz->BlueMin     = 0.0; //...
	viz->BlueMax     = 5.0; //...
	
	//Create a GeoTag - this is not required, but included for illustration. In this example, we know the location and time this image was taken, but not the
	//orientation of the camera.
	FRFGeoTag GeoTag;
	GeoTag.P_ECEF << 2500.0, 3871.1, -3861.3;
	GeoTag.C_ECEF_Cam << std::nan(""), std::nan(""), std::nan(""),
	                     std::nan(""), std::nan(""), std::nan(""),
	                     std::nan(""), std::nan(""), std::nan("");
	GeoTag.GPST_Week = 1965U;
	GeoTag.GPST_TOW  = 286337.72;
	myImage.SetGeoTag(GeoTag);
	
	//We now have a minimal FRF image. Save it.
	myImage.SaveToDisk(Filename);
}

static void printLayerInfo(FRFLayer * Layer) {
	fprintf(stderr, "Name: %s\r\n", Layer->Name.c_str());
	fprintf(stderr, "Description: %s\r\n", Layer->Description.c_str());
	fprintf(stderr, "Units: %s\r\n", Layer->GetUnitsStringInMostReadableForm().c_str());
	fprintf(stderr, "Type Code: %u (%s)\r\n", (unsigned int) Layer->GetTypeCode(), Layer->GetTypeString().c_str());
	fprintf(stderr, "alpha: %.8e,  beta: %.8f\r\n", Layer->alpha, Layer->beta);
	fprintf(stderr, "Validity Mask: %s\r\n", Layer->HasValidityMask ? "Yes" : "No");
	fprintf(stderr, "Total size: %f MB\r\n", ((double) (Layer->GetTotalBytesWithValidityMask() / ((uint64_t) 100000U)))/10.0);
	
	std::tuple<double, double> range = Layer->GetRange();
	if (Layer->GetTypeCode() <= 68U) {
		fprintf(stderr, "Value Range: [%f, %f]\r\n", std::get<0>(range), std::get<1>(range));
		fprintf(stderr, "Value Step Size: %e\r\n", Layer->GetStepSize());
	}
	else
		fprintf(stderr, "Value Range: [%e, %e]\r\n", std::get<0>(range), std::get<1>(range));
}

static void LoadImageAndInspectPixels(std::string Filename) {
	FRFImage myImage; //Create a new empty FRF Image
	
	//Clear the image object and re-load from disk
	if (myImage.LoadFromDisk(Filename))
		std::cerr << "Image loaded successfully.\r\n";
	else
		std::cerr << "Image loading failed.\r\n";
	
	std::tuple<uint16_t, uint16_t> fileVersion = myImage.getFileVersion();
	fprintf(stderr,"FRF version %u.%u\r\n", (unsigned int) std::get<0>(fileVersion), (unsigned int) std::get<1>(fileVersion));
	fprintf(stderr,"Dimensions: %u rows x %u cols\r\n", (unsigned int) myImage.Height(), (unsigned int) myImage.Width());
	fprintf(stderr,"Number of Layers: %u,  Number of Visualizations: %u\r\n", (unsigned int) myImage.NumberOfLayers(), (unsigned int) myImage.GetNumberOfVisualizations());
	
	//Print out layer info for each layer
	for (uint16_t layerIndex = 0U; layerIndex < myImage.NumberOfLayers(); layerIndex++) {
		fprintf(stderr,"\r\n**************  Layer %u  **************\r\n", (unsigned int) layerIndex);
		printLayerInfo(myImage.Layer(layerIndex));
	}
	std::cerr << "\r\n";
	
	//Print out some pixel values
	if (myImage.NumberOfLayers() == 0U)
		std::cerr << "Uh-oh. This image is empty. Nothing to display.\r\n";
	else {
		std::cerr << "Value at (73,24): " << myImage.Layer(0U)->GetValue(73U, 24U) << "\r\n";
		std::cerr << "Value at (12,82): " << myImage.Layer(0U)->GetValue(12U, 82U) << "\r\n";
		std::cerr << "Value at (31,3900): " << myImage.Layer(0U)->GetValue(31U, 3900U) << "\r\n";
		std::cerr << "Value at (19,122): " << myImage.Layer(0U)->GetValue(19U, 122U) << "\r\n";
		std::cerr << "Value at (19,123): " << myImage.Layer(0U)->GetValue(19U, 123U) << "\r\n";
	}
	std::cerr << "\r\n";
}

int main(int argc, char* argv[]) {
	std::string Filename("MyImage.frf");
	CreateImageAndSaveIt(Filename);
	
	std::cerr << "\r\nLoading Image\r\n\r\n";
	LoadImageAndInspectPixels(Filename);
	
	return 0;
}







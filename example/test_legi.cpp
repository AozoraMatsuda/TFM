// sample code for simulating LEGI based migration and shape control of cells without focal adhesions
// needs minor completion and adaptation
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <time.h>
#define MINRAD 10.0
typedef struct tagBOUNDARY
{
	float fAngle;  // angular position using cell center as the origin
	float fRadius; // distance from the cell center
	float fSignal; // position dependent protrusion signals
} BOUNDARY;

void SimulateMigration(void)
{
	// generic variables for computation
	int i, j, k, n;
	float fData1, fData2, fData3, fData4, fData5, fData6;
	double dData;

	// 10 input variables for the model cell
	int nTimeScale;
	float fDiffuse, fDecay, fRetractionRate, fProtrusionRate;
	float fBurstRate, fBurstSize, fFeedbackSlope, fFeedbackOffPoint;
	float fInhibitorConc;

	// model state
	BOUNDARY bound1[362], bound2[362]; // cell perimeter, one point per degree, at two consecutive time points
	float fArea, fRadMax;
	float fDistance[360]; // distance between neighboring perimeter points
	float fX, fY;		  // position of cell center
	float fDX, fDY;		  // displacement of cell center at each iteraction
	struct
	{
		float fx;
		float fy;
	} Pos1, Pos2, Pos3, StartPos = {0, 0}; // positions of cell center at 3 consecutive time points
										   // and position where behavior measurement starts
	int nCurrentCycle = 1;				   // current iteration cycles number
	float fInhibitor;					   // retraction signals

	float fSignalMin = (float)pow(10, -4.0); // lower limit of protrusion signals
	float fSignalMax = (float)100.0;		 // upper limit of protrusion signals
	int nStartAnalysis;						 // iteration cycle to start behavior measurements
	int nSample = 0;						 // sample number for behavior statistics

	// variable for persistence measurements
	int nPersistLength;			// user defined reference length
	float fPersistX, fPersistY; // starting location for timing the migration over a reference length
	int nPersistTime;			// number of iterative cycles it took to migrate over a reference length
	int nPersistTimeSum;		// accumulator for nPersistTime, for calculating the average
	int nPersist;				// how many times the cell has migrated over the reference length

	// variables for behavior statistics
	double dRadMax = 0.0, dRadMaxSq = 0.0;
	double dArea = 0.0, dAreaSq = 0.0;
	double dSpeed = 0.0, dSpeedSq = 0.0;
	double dPersistence = 0.0, dPersistenceSq = 0.0;
	double dRoundness = 0.0, dRoundnessSq = 0.0;

	gsl_rng *rng; // random number generator using the OpenSource gsl package

	// complete these lines by setting the value of each input variable
	fDiffuse = ;		  // diffusion constant of stimulus along the membrane
	fDecay = ;			  // percentage decay of stimulus per cycle
	fBurstRate = ;		  // burst rate
	fInhibitorConc = ;	  // inhibitor concentration per pixel
	fRetractionRate = ;	  // retraction rate
	fProtrusionRate = ;	  // protrusion rate
	fBurstSize = ;		  // burst size
	fFeedbackSlope = ;	  // feedback curve slope
	fFeedbackOffPoint = ; // feedback curve take off point
	nTimeScale = ;		  // sampling interval for behavior measurements

	nStartAnalysis = ;
	nPersistLength = ;
	//

	fDecay = (float)(1.0 - fDecay);
	rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng, (unsigned)time(NULL));

	// set starting position and shape
	fX = 256.0;
	fY = 256.0;
	fArea = (float)(M_PI * MINRAD * MINRAD);
	//

	// calculate initial protrusion signals assuming pseudo-steady state
	fData2 = (float)(360.0 * fArea * fInhibitorConc);
	fData1 = (float)((fBurstSize * fBurstRate - fFeedbackSlope * fFeedbackOffPoint * fBurstSize) / (1.0 - fDecay - fFeedbackSlope * (1.0 - fData2) * fBurstSize));
	if (fData2 >= 1.0 || fData1 * (1.0 - fData2) < fFeedbackOffPoint)
		fData1 = (float)(fBurstSize * fBurstRate / (1.0 - fDecay));
	if (fData1 > fSignalMax)
		fData1 = fSignalMax;

	for (i = 0; i < 360; ++i)
	{
		bound1[i].fAngle = (float)i;
		bound1[i].fRadius = MINRAD;
		bound1[i].fSignal = fData1;
	}
	bound1[360] = bound1[0];
	bound1[361] = bound1[1];

	fInhibitor = (float)(fData1 * 360 * fInhibitorConc * fArea);

	// start iterative simulation cycles
	for (;;) // infinite iterations for now; need to implement a stopper
	{
		fData3 = 0.0; // accumulate total protrusion signals

		fData4 = (float)sqrt(fDiffuse * 2.0);
		// calculate the distance between neighboring perimeter points
		for (i = 0; i < 360; ++i)
			fDistance[i] = (float)sqrt(bound1[i].fRadius * bound1[i].fRadius + bound1[i + 1].fRadius * bound1[i + 1].fRadius - 2.0 * bound1[i].fRadius * bound1[i + 1].fRadius * cos((bound1[i].fAngle - bound1[i + 1].fAngle) * M_PI / 180.0));

		bound2[0] = bound1[0];
		bound2[361] = bound1[361];
		for (i = 1; i <= 360; ++i)
		{
			// diffusion and decay
			bound2[i] = bound1[i];
			fData5 = 0;
			fData6 = 0;
			n = 0;
			for (j = i; j < i + 180; ++j)
			{
				if (j >= 360)
					k = j - 360;
				else
					k = j;
				fData5 = fData5 + fDistance[k];
				fData6 = fData6 + bound1[k + 1].fSignal;
				n = n + 1;
				if (fData5 > fData4)
					break;
			}
			fData6 = fData6 / n;
			if (n == 1)
				fData1 = fDiffuse / fData5 / fData5;
			else
				fData1 = 0.5;
			fData1 = (float)((fData6 - bound1[i].fSignal) * fData1);

			fData5 = 0;
			fData6 = 0;
			n = 0;
			for (j = i + 359; j >= i + 180; --j)
			{
				if (j >= 360)
					k = j - 360;
				else
					k = j;
				fData5 = fData5 + fDistance[k];
				fData6 = fData6 + bound1[k].fSignal;
				n = n + 1;
				if (fData5 > fData4)
					break;
			}
			fData6 = fData6 / n;
			if (n == 1)
				fData2 = fDiffuse / fData5 / fData5;
			else
				fData2 = 0.5;
			fData2 = (float)((fData6 - bound1[i].fSignal) * fData2);

			fData1 = (fData1 + fData2) / 2;
			bound2[i].fSignal = (float)(bound1[i].fSignal + fData1) * fDecay;

			// new protrusion signals from positive feedback
			fData1 = (float)((bound1[i].fSignal - fInhibitor - fFeedbackOffPoint) * fFeedbackSlope);
			if (fData1 < 0.0)
				fData1 = 0.0;
			fData1 = fData1 + fBurstRate;

			fData1 *= fBurstSize;
			fData1 = (float)Gaussian(fData1, fData1);
			if (fData1 > 0.0)
				bound2[i].fSignal += fData1;

			if (bound2[i].fSignal < fSignalMin)
				bound2[i].fSignal = 0.0;
			else if (bound2[i].fSignal > fSignalMax)
				bound2[i].fSignal = fSignalMax;

			// total protrusion signals
			fData3 = (float)(fData3 + bound2[i].fSignal);
		}
		bound2[0] = bound2[360];
		bound2[361] = bound2[1];

		fInhibitor = (float)(fData3 * fInhibitorConc * fArea);

		// change cell shape according to the signal
		// retraction
		for (i = 0; i < 360; ++i)
		{
			if (bound2[i].fSignal <= fInhibitor)
			{
				if (bound2[i].fRadius > MINRAD)
					bound2[i].fRadius -= (float)((bound2[i].fRadius - MINRAD) * fRetractionRate);
				if (bound2[i].fRadius < MINRAD)
					bound2[i].fRadius = MINRAD;
			}
		}
		// protrusion
		for (i = 0; i < 360; ++i)
		{
			if (bound2[i].fSignal > fInhibitor)
			{
				fData1 = Gaussian(fProtrusionRate, (float)fProtrusionRate);
				if (fData1 > 0.0)
				{
					bound2[i].fRadius = bound2[i].fRadius + fData1;
				}
			}
		}
		bound2[360] = bound2[0];
		bound2[361] = bound2[1];

		// calculate the new cell center
		fDX = 0.0;
		fDY = 0.0;
		for (i = 0; i < 360; ++i)
		{
			fDX = (float)(fDX + bound2[i].fRadius * cos(bound2[i].fAngle * M_PI / 180.0));
			fDY = (float)(fDY + bound2[i].fRadius * sin(bound2[i].fAngle * M_PI / 180.0));
		}
		fDX = fDX / 360;
		fDY = fDY / 360;
		fX = fX + fDX;
		fY = fY + fDY;

		// calculate polar coordinates relative to the new center
		for (i = 0; i < 360; ++i)
		{
			fData1 = (float)(bound2[i].fRadius * cos(bound2[i].fAngle * M_PI / 180.0) - fDX);
			fData2 = (float)(bound2[i].fRadius * sin(bound2[i].fAngle * M_PI / 180.0) - fDY);
			bound2[i].fRadius = (float)sqrt(fData1 * fData1 + fData2 * fData2);
			bound2[i].fAngle = Angle(fData1, fData2);
		}
		qsort((void *)bound2, 360, sizeof(BOUNDARY), CmpAngle);

		bound2[360] = bound2[0];
		bound2[0] = bound2[359];
		bound2[0].fAngle = bound2[0].fAngle - 360;
		for (i = 359; i > 1; --i)
			bound2[i] = bound2[i - 1];
		bound2[1] = bound2[360];

		bound2[360].fAngle = bound2[360].fAngle + 360;
		bound2[361] = bound2[1];
		bound2[361].fAngle = bound2[361].fAngle + 360;

		// make sure there is one perimeter point per degree
		for (i = 0; i < 360; ++i)
		{
			for (j = 0; j <= 360; ++j)
			{
				if (bound2[j].fAngle + 0.5 >= i && bound2[j].fAngle + 0.5 < i + 1)
				{
					bound1[i].fAngle = bound2[j].fAngle;
					bound1[i].fRadius = bound2[j].fRadius;
					bound1[i].fSignal = bound2[j].fSignal;
					break;
				}
				else if (bound2[j].fAngle <= i && bound2[j + 1].fAngle >= i)
				{ // interpolation
					bound1[i].fAngle = (float)i;
					bound1[i].fRadius = bound2[j].fRadius +
										(bound2[j + 1].fRadius - bound2[j].fRadius) * ((float)i - bound2[j].fAngle) / (bound2[j + 1].fAngle - bound2[j].fAngle);
					bound1[i].fSignal = bound2[j].fSignal +
										(bound2[j + 1].fSignal - bound2[j].fSignal) * ((float)i - bound2[j].fAngle) / (bound2[j + 1].fAngle - bound2[j].fAngle);
					break;
				}
			}
		}
		bound1[360] = bound1[0];
		bound1[361] = bound1[1];

		// do behavior measurements
		fArea = 0.0;
		for (i = 0; i < 360; ++i)
			fArea = fArea + bound1[i].fRadius * bound1[i].fRadius;
		fArea = (float)(fArea * M_PI / 360.0);

		if (nCurrentCycle == nStartAnalysis) // starting time for behavior measurements
		{
			StartPos.fx = fX;
			StartPos.fy = fY;
			Pos1.fx = fX;
			Pos1.fy = fY;
			Pos2 = Pos1;
			Pos3 = Pos1;
			fPersistX = fX;
			fPersistY = fY;
			nPersistTimeSum = 0;
			nPersistTime = 0;
			nPersist = 0;
		}
		else if (nCurrentCycle > nStartAnalysis && nCurrentCycle % nTimeScale == 0)
		{ // make behavior measurements every nTimeScale iterative cycles
			Pos1 = Pos2;
			Pos2 = Pos3;
			Pos3.fx = fX;
			Pos3.fy = fY;
			nSample += 1;
			fRadMax = 0.0;
			for (i = 0; i < 360; ++i)
				if (bound1[i].fRadius > fRadMax)
					fRadMax = bound1[i].fRadius;
			dRadMax += fRadMax;				// for calculating average
			dRadMaxSq += fRadMax * fRadMax; // for calculating standard deviation

			dArea += fArea;
			dAreaSq += fArea * fArea;

			dData = pow(Pos3.fx - Pos2.fx, 2) + pow(Pos3.fy - Pos2.fy, 2);
			dSpeedSq += dData;
			dSpeed += sqrt(dData);

			dData = fArea / (M_PI * pow(fRadMax, 2));
			dRoundness += dData;
			dRoundnessSq += dData * dData;

			// persistence calculation
			fData1 = fX - fPersistX;
			fData2 = fY - fPersistY;
			if (fData1 * fData1 + fData2 * fData2 >= nPersistLength * nPersistLength)
			{ // finished moving one reference length
				fPersistX = fX;
				fPersistY = fY;
				nPersistTimeSum += nPersistTime;
				nPersistTime = 0;
				nPersist += 1;
			}
			else
				nPersistTime += 1;
		}
		++nCurrentCycle;
	}
	if (nSample > 1)
	{
		// calculate and output average behavior metrics with standard deviation
		// for example maximal radius, area, speed, net distance, persistence, roundness
		// show here only the calculation of speed and persistence
		dSpeed = dSpeed / nSample;
		if (nPersistTimeSum == 0 || dSpeed == 0)
			dPersistence = 0.0;
		else
		{
			if (nPersistTime * nPersist > nPersistTimeSum)
			{ // include the last, incomplete measurement of PersistTime if it exceeds the average
				nPersistTimeSum += nPersistTime;
				++nPersist;
			}
			dPersistence = (nPersist * nPersistLength) / (nPersistTimeSum * dSpeed);
		}
	}
	if (rng != NULL)
		gsl_rng_free(rng);
	return;
}

float Angle(float fX, float fY)
{ // return angular position according to the x, y coordinates
	float fAngle;

	if (fX == 0)
	{
		if (fY == 0)
			return 0.0;
		else if (fY > 0)
			return 90.0;
		else
			return 270.0;
	}
	fAngle = (float)atan2(fY, fX);
	fAngle = (float)(fAngle * (180.0 / M_PI));
	if (fAngle < 0.0)
		fAngle = (float)(fAngle + 360.0);
	return fAngle;
}

int CmpAngle(const void *bndpt1, const void *bndpt2)
{ // callback function for qsort, for sorting perimeter points according to their angular positions
	float fAngle1, fAngle2;

	fAngle1 = ((BOUNDARY *)bndpt1)->fAngle;
	fAngle2 = ((BOUNDARY *)bndpt2)->fAngle;
	if (fAngle1 < fAngle2)
		return (-1);
	else if (fAngle1 == fAngle2)
		return (0);
	else
		return (1);
}

float Gaussian(float fAverage, float fVariance)
{ // generate random numbers that fit a Gaussian distribution with specified average and variance
	static gsl_rng *rng = NULL;

	if (rng == NULL)
	{
		rng = gsl_rng_alloc(gsl_rng_taus);
		gsl_rng_set(rng, (unsigned)time(NULL));
	}
	return ((float)gsl_ran_gaussian(rng, sqrt(fVariance)) + fAverage);
}

\name{SAVE}
\alias{SAVE}
\alias{show.SAVE}
\alias{show,SAVE-method}
\alias{show,summary.SAVE-method}
\alias{summary.SAVE}
\alias{summary.SAVE-class}
\alias{summary,SAVE-method}
\alias{show.summary.SAVE}
\title{ Setting up the analysis of a computer model }
\description{ Setting up the SAVE methodology: construction of an emulator of the computer model and estimation of the parameters of the Gaussian process for the bias function. }
\usage{
SAVE(response.name=NULL, controllable.names=NULL, calibration.names=NULL, 
	field.data=NULL, model.data=NULL, mean.formula=~1, bestguess=NULL,
	kriging.controls=SAVE.controls(), verbose=FALSE)

\S4method{show}{SAVE}(object)

\S4method{summary}{SAVE}(object)

\S4method{show}{summary.SAVE}(object)
}
\arguments{
  \item{response.name}{A \code{character} object with the name of the response variable.}
 \item{controllable.names}{Either a \code{character} object with the
names of the controllable inputs or set to NULL if in the field experiment
the controllable inputs (if any) have not been varied (the analysis has
constant controllable inputs).}
  \item{calibration.names}{A \code{character} object with the names of the calibration inputs. Use NULL if the model does not depend on any calibration inputs.}
  \item{field.data}{A \code{data.frame} with the observations coming from the field experiment. The column names of this \code{data.frame} should contain the \code{controllable.names} and the \code{response.name}.}
  \item{model.data}{A \code{data.frame} with the observations coming from runs of the computer model. The column names of this \code{data.frame} should contain the \code{controllable.names}, the \code{calibration.names} and the \code{response.name}.}
  \item{mean.formula}{A \code{formula} specifying the mean function of the Gaussian Process approximation to the output of the computer model (the emulator). Can only involve terms in \code{controllable.names}.}
  \item{bestguess}{A named \code{list} specifying the best guess for the calibration parameters.}
  \item{kriging.controls}{A named \code{list} specifying the parameters to be passed to the kriging process at the stage I parameters (the ones invoved in the construction of the emulator) estimation step.}
  \item{verbose}{A \code{logical} value indicating the level of output as the function runs.}
  \item{object}{An object of the corresponding signature.}
}  
     
\details{ Based on computer model runs, \code{SAVE} fits an approximation of the model output, usually called an emulator. The emulator is constructed using the Gaussian process response technique (GASP), described in more detail in \code{\link[SAVE]{SAVE-class}}. Further, at this stage an estimation of the parameters of the Gaussian process specifying the bias function (difference between field observations and computer model outputs) is also performed. Some of the calculations are done using the package \code{DiceKriging}}

\value{
  \code{SAVE} returns an S4 object of class \code{SAVE} (see \code{SAVE-class}). 
}

\references{
Palomo J, Paulo R, Garcia-Donato G (2015). SAVE: An R Package for the Statistical Analysis of Computer Models. Journal of Statistical Software, 64(13), 1-23. Available from http://www.jstatsoft.org/v64/i13/
        
Bayarri MJ, Berger JO, Paulo R, Sacks J, Cafeo JA, Cavendish J, Lin CH, Tu J (2007). A Framework for Validation of Computer Models. Technometrics, 49, 138-154.

Craig P, Goldstein M, Seheult A, Smith J (1996). Bayes linear strategies for history matching of hydrocarbon reservoirs. In JM Bernardo, JO Berger, AP Dawid, D Heckerman, AFM Smith (eds.), Bayesian Statistics 5. Oxford University Press: London. (with discussion).

Higdon D, Kennedy MC, Cavendish J, Cafeo J, Ryne RD (2004). Combining field data and computer simulations for calibration and prediction. SIAM Journal on Scientific Computing, 26, 448-466.

Kennedy MC, O Hagan A (2001). Bayesian calibration of computer models (with discussion). Journal of the Royal Statistical Society B, 63, 425-464.

Roustant O., Ginsbourger D. and Deville Y. (2012).
  DiceKriging, DiceOptim: Two R Packages for the Analysis of Computer
  Experiments by Kriging-Based Metamodeling and Optimization. Journal
  of Statistical Software, 51(1), 1-55.
  
}


\author{Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato}
  
\seealso{ \code{\link[=predictcode]{predictcode}},
	  	  \code{\link[=bayesfit.SAVE]{bayesfit}}  
}
	
\examples{
\dontrun{
library(SAVE)

#############
# load data
#############

data(spotweldfield,package='SAVE')
data(spotweldmodel,package='SAVE')

##############
# create the SAVE object which describes the problem and
# compute the corresponding mle estimates
##############

gfsw <- SAVE(response.name="diameter", controllable.names=c("current", "load", "thickness"),
			 calibration.names="tuning", field.data=spotweldfield,
			 model.data=spotweldmodel, mean.formula=~1,
			 bestguess=list(tuning=4.0))

# summary of the results

summary(gfsw)
}

}



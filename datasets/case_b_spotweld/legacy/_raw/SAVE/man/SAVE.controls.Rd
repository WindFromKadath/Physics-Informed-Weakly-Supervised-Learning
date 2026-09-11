\name{SAVE.controls}
\alias{SAVE.controls}
\title{ Control parameters for kriging process. }
\description{ Returns a list suitable as the \code{kriging.controls} argument of the \code{SAVE} function.}
\usage{SAVE.controls(lower = NULL, upper = NULL, optim.method = "BFGS", parinit = NULL,...)}
\arguments{
  \item{lower}{(see below)}
  \item{upper}{optional vector containing the bounds of the correlation
          parameters for optimization in function \code{\link[DiceKriging]{km}} in package \code{DiceKriging}. The default values are given by \code{\link[DiceKriging]{covParametersBounds}}.}
  \item{optim.method}{an optional character string indicating which
          optimization method is chosen for the likelihood
          maximization. \code{"BFGS"} is the \code{optim} quasi-Newton procedure
          of package \code{stats}, with the method "L-BFGS-B". \code{"gen"} is
          the \code{genoud} genetic algorithm (using derivatives) from
          package \code{rgenoud} (>= 5.3.3).}
  \item{parinit}{an optional vector containing the initial values for the
          variables to be optimized over. If no vector is given, an
          initial point is generated as follows. For method \code{"gen"},
          the initial point is generated uniformly inside the
          hyper-rectangle domain defined by \code{lower} and \code{upper}. For
          method \code{"BFGS"}, some points (see \code{control} below) are
          generated uniformly in the domain. Then the best point with
          respect to the likelihood (or penalized likelihood, see
          \code{penalty}) criterion is chosen.}
  \item{\dots}{arguments for \code{SAVE.controls} if they are not given explicitly. Current version deprecates any other argument different than those listed above.}
}
\value{	a list of parameters that are allowed to be passed to the \code{\link[DiceKriging]{km}} function in package \code{\link[DiceKriging]{DiceKriging}}.
  \item{lower}{(see above)}
  \item{upper}{(see above)}
  \item{optim.method}{(see above)}
  \item{parinit}{(see above)}
}
\author{
Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato
}
\seealso{See Also \code{\link[DiceKriging]{DiceKriging}} for more details on the parametes.}
\examples{
\dontrun{
	library(SAVE)
	sc  <- SAVE.controls(lower = 0.2, upper = 0.7, optim.method = "BFGS")
	sc1 <- SAVE.controls(lower = 0.2, upper = 0.7, optim.method = "BFGS",
				coef.var=1, coef.trend=0)
	### In this case, the parameters \code{coef.var} and \code{coef.trend} 
	### will be deprecated since \code{\link{SAVE}} does not allow to fix 
	### any parameter different than: \code{lower}, \code{upper}, 
	### \code{optim.method} and \code{parinit}.
 }
}
\keyword{ internal }

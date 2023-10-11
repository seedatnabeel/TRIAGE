import numpy as np
import pandas as pd

from crepes import ConformalRegressor, ConformalPredictiveSystem

from crepes.fillings import (sigma_variance, 
                             sigma_variance_oob,
                             sigma_knn,
                             binning)

class Triage:
     
    def __init__(self, X_eval, y_eval, X_cal, y_cal, learner):
            self.X_eval = X_eval
            self.y_eval = y_eval
            self.X_cal = X_cal
            self.y_cal = y_cal
            self.learner = learner

            self.triage_score_array = None
            self.interval_array = None
            self.crps_array = None
            self.cpds_array = None
            self.errors_array = None
            self.preds_array = None
            self.upper_array = None
            self.lower_array = None

    def run(self, metric='triage', percentile_thresh = 75, conf_thresh = 0.33, compute_cpd=False, compute_crps=False, learner=None):

            if learner  != None:
                self.learner = learner

            triage_score_array, interval_array, crps_array, cpds_array, errors_array, preds_array, upper_array, lower_array = self.compute_scoring_metrics(compute_cpd=compute_cpd, compute_crps=compute_crps)
            

            conf_thresh_low = conf_thresh
            conf_thresh_high = 1 - conf_thresh

            if metric == 'triage':
                score_metric = triage_score_array
            elif metric == 'interval':
                score_metric = interval_array
            elif metric == 'crps':
                score_metric = crps_array
            elif metric == 'cpds':
                score_metric = cpds_array
            elif metric == 'errors':
                score_metric = errors_array
            elif metric == 'preds':
                score_metric = preds_array
                
            uncertainty = np.std(score_metric, axis=-1) 
            confidence = np.mean(score_metric, axis=-1)

            # over-estimated
            oe_group = np.where(
            (confidence <= conf_thresh_low)
            & (uncertainty <= np.percentile(uncertainty, percentile_thresh))
            )[0]

            # under-estimated
            ue_group = np.where(
            (confidence >= conf_thresh_high)
            & (uncertainty <= np.percentile(uncertainty, percentile_thresh))
            )[0]

            ue_oe_group = np.concatenate((ue_group, oe_group))

            # well estimated
            we_group = []
            for id in range(len(confidence)):
                if id not in ue_oe_group:
                    we_group.append(id)

            
            we_group = np.array(we_group)


    
            groups = []
            for i in range(len(score_metric)):
                if i in ue_group:
                    groups.append(0)
                if i in we_group:
                    groups.append(1)
                if i in oe_group:
                    groups.append(2)

            groups = np.array(groups)

            groups_ids = { 
                "ue_group": ue_group,
                "oe_group": oe_group,
                "we_group": we_group,
            }

            raw_metrics = {
                "variability": uncertainty,
                    "confidence": confidence,
                    "score_metric": score_metric,
            }


            return groups_ids, raw_metrics


    def eval_crps(self, model, y_hat=None, y=None, sigmas=None, bins=None, alphas=None, normalized=True,
                    confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics="CRPS"):
            """
            Evaluate conformal predictive system.

            Parameters
            ----------
            y_hat : array-like of shape (n_values,), default=None,
                predicted values
            y : array-like of shape (n_values,), default=None,
                correct target values
            sigmas : array-like of shape (n_values,), default=None,
                difficulty estimates
            bins : array-like of shape (n_values,), default=None,
                Mondrian categories
            confidence : float in range (0,1), default=0.95
                confidence level
            y_min : float or int, default=-np.inf
                minimum value to include in prediction intervals
            y_max : float or int, default=np.inf
                maximum value to include in prediction intervals
            metrics : a string or a list of strings, default=list of all 
                metrics; ["error", "efficiency", "CRPS", "time_fit", 
                        "time_evaluate"]
            
            Returns
            -------
            results : dictionary with a key for each selected metric 
                estimated performance using the metrics
            """

            if metrics is None:
                metrics = ["error","efficiency","CRPS","time_fit","time_evaluate"]
            lower_percentile = (1-confidence)/2*100
            higher_percentile = (confidence+(1-confidence)/2)*100
            test_results = {}
            
            results, cpds = model.predict(y_hat, sigmas=sigmas, bins=bins, y=y,
                                        lower_percentiles=lower_percentile,
                                        higher_percentiles=higher_percentile,
                                        y_min=y_min, y_max=y_max,
                                        return_cpds=True)
            intervals = results[:,[1,2]]
            
            if normalized:
                crps = self.calculate_crps(cpds, alphas, sigmas, y)
            else:
                crps = self.calculate_crps(cpds, alphas,
                                    np.ones(len(y_hat)), y)

            return crps
            
    def calculate_crps(self, cpds, alphas, sigmas, y):
        """
        Calculate mean continuous-ranked probability score (crps)
        for a set of conformal predictive distributions.

        Parameters
        ----------
        cpds : array-like of shape (n_values, c_values)
            conformal predictive distributions
        alphas : array-like of shape (c_values,)
            sorted (normalized) residuals of the calibration examples 
        sigmas : array-like of shape (n_values,),
            difficulty estimates
        y : array-like of shape (n_values,)
            correct target values
            
        Returns
        -------
        crps : float
            mean continuous-ranked probability score for the conformal
            predictive distributions 
        """
        widths = np.array([alphas[i+1]-alphas[i] for i in range(len(alphas)-1)])
        cum_probs = np.cumsum([1/len(alphas) for i in range(len(alphas)-1)])
        lower_errors = cum_probs**2
        higher_errors = (1-cum_probs)**2
        cpd_indexes = [np.argwhere(cpds[i]<y[i]) for i in range(len(y))]
        cpd_indexes = [-1 if len(c)==0 else c[-1][0] for c in cpd_indexes]
        return np.array([self.get_crps(cpd_indexes[i], lower_errors, higher_errors,
                                widths, sigmas[i], cpds[i], y[i])
                        for i in range(len(y))])

    def get_crps(self, cpd_index, lower_errors, higher_errors, widths, sigma, cpd, y):
        """
        Calculate continuous-ranked probability score (crps) for a single
        conformal predictive distribution. 

        Parameters
        ----------
        cpd_index : int
            highest index for which y is higher than the corresponding cpd value
        lower_errors : array-like of shape (c_values-1,)
            values to add to crps for values less than y
        higher_errors : array-like of shape (c_values-1,)
            values to add to crps for values higher than y
        widths : array-like of shape (c_values-1,),
            differences between consecutive pairs of sorted (normalized) residuals 
            of the calibration examples 
        sigma : int or float
            difficulty estimate for single object
        cpd : array-like of shape (c_values,)
            conformal predictive distyribution
        y : int or float
            correct target value
            
        Returns
        -------
        crps : float
            continuous-ranked probability score
        """
        if cpd_index == -1:
            score = np.sum(higher_errors*widths*sigma)+(cpd[0]-y) 
        elif cpd_index == len(cpd)-1:
            score = np.sum(lower_errors*widths*sigma)+(y-cpd[-1]) 
        else:
            score = np.sum(lower_errors[:cpd_index]*widths[:cpd_index]*sigma) +\
                np.sum(higher_errors[cpd_index+1:]*widths[cpd_index+1:]*sigma) +\
                lower_errors[cpd_index]*(y-cpd[cpd_index])*sigma +\
                higher_errors[cpd_index]*(cpd[cpd_index+1]-y)*sigma
        return score



    def compute_pred_metrics(self, iteration=10, knn=True):

        y_hat_cal = self.learner(self.X_cal) 
        y_hat_cal = y_hat_cal.detach().cpu().numpy().reshape(-1)
        y_cal = self.y_cal.detach().cpu().numpy().reshape(-1)
        y_cal, y_hat_cal = np.squeeze(y_cal), np.squeeze(y_hat_cal)
    
        residuals_cal = y_cal - y_hat_cal

        y_hat_eval = self.learner(self.X_eval) 
        y_hat_eval = y_hat_eval.detach().cpu().numpy().reshape(-1)


        if knn==True:
            sigmas_cal = sigma_knn(X=self.X_cal.detach().cpu().numpy(), residuals=residuals_cal)
            cr_norm = ConformalRegressor()
            cr_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)

            sigmas_test = sigma_knn(X=self.X_cal.detach().cpu().numpy(), residuals=residuals_cal, X_test=self.X_eval.detach().cpu().numpy())


        else:
            sigmas_cal = sigma_variance(X=self.X_cal.detach().cpu().numpy(), learner=self.learner)

            cr_norm = ConformalRegressor()
            cr_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)

            sigmas_test = sigma_variance(X=self.X_eval.detach().cpu().numpy(), learner=self.learner)


        intervals_norm = cr_norm.predict(y_hat=y_hat_eval, 
                                                    sigmas=sigmas_test,
                                                    y_min=0, y_max=1)
        
        return residuals_cal, sigmas_cal, sigmas_test, y_hat_eval



    def compute_scoring_metrics(self, compute_cpd=False, compute_crps=False):


        self.learner.eval()
        y_hat = self.learner(self.X_eval) 
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
        y_eval = self.y_eval.detach().cpu().numpy().reshape(-1)
        y_eval, y_hat = np.squeeze(y_eval), np.squeeze(y_hat)

        # compute the errors
        errors = y_eval - y_hat
        errors_values = np.expand_dims(errors,1)

        residuals_cal, sigmas_cal, sigmas, y_hat =  self.compute_pred_metrics()
 
        # compute normalized conformal predictive system
        cps_norm = ConformalPredictiveSystem().fit(residuals=residuals_cal,
                                            sigmas=sigmas_cal)


        triage_score = cps_norm.predict(y_hat=y_hat,
                                    sigmas=sigmas,
                                    y=self.y_eval.detach().cpu().numpy().reshape(-1))


        # INTERVALS
        intervals = cps_norm.predict(y_hat=y_hat, 
                                sigmas=sigmas, 
                                lower_percentiles=5, 
                                higher_percentiles=95)

        interval_width = np.array(intervals[:,1]-intervals[:,0])
        interval_values = np.expand_dims(interval_width,1)
        
        upper = np.expand_dims(np.array(intervals[:,1]),1)
        lower = np.expand_dims(np.array(intervals[:,0]),1)

        if compute_cpd == True:

            # CPDS
            cpds_values = cps_norm.predict(y_hat=y_hat,
                                    sigmas=sigmas,
                                    return_cpds=True) 
            
            if self.cpds_array is None:  # On first epoch of training
                self.cpds_array = cpds_values  
            else:
                self.cpds_array = np.dstack((self.cpds_array, cpds_values))

        if compute_crps == True:
            # CRPS
            crps = self.eval_crps(model =cps_norm, y_hat=y_hat, y=self.y_eval, sigmas=sigmas, bins=None, alphas=np.sort(residuals_cal/sigmas_cal),
                    confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics="CRPS")
            crps_values = np.expand_dims(crps,1)

            if self.crps_array is None:  # On first epoch of training
                self.crps_array = crps_values  
            else:
                stack = [
                    self.crps_array,
                    crps_values,
                ]
                self.crps_array = np.hstack(stack)


        # stack metrics       
        if self.upper_array is None:  # On first epoch of training
                self.upper_array = upper 
        else:
            stack = [
                self.upper_array,
                upper,
            ]
            self.upper_array = np.hstack(stack)
        
        if self.lower_array is None:  # On first epoch of training
                self.lower_array = lower
        else:
            stack = [
                self.lower_array,
                lower,
            ]
            self.lower_array = np.hstack(stack)
        
        if self.preds_array is None:  # On first epoch of training
                self.preds_array = y_hat  
        else:
            stack = [
                self.preds_array,
                y_hat,
            ]
            self.preds_array = np.hstack(stack)

        if self.triage_score_array is None:  # On first epoch of training
                self.triage_score_array = triage_score
        else:
            stack = [
                self.triage_score_array,
                triage_score,
            ]
            self.triage_score_array = np.hstack(stack)



        if self.interval_array is None:  # On first epoch of training
                self.interval_array = interval_values  
        else:
            stack = [
                self.interval_array,
                interval_values,
            ]
            self.interval_array = np.hstack(stack)


        if self.errors_array is None:  # On first epoch of training
                self.errors_array = errors_values
        else:
            stack = [
                self.errors_array,
                errors_values,
            ]
            self.errors_array = np.hstack(stack)

        return self.triage_score_array, self.interval_array, self.crps_array, self.cpds_array, self.errors_array, self.preds_array, self.upper_array, self.lower_array


    @property
    def _triage_score_array(self) -> np.ndarray:
        return self.triage_score_array
    
    @property
    def _interval_array(self) -> np.ndarray:
        return self.interval_array
    
    @property
    def _crps_array(self) -> np.ndarray:
        return self.crps_array
    
    @property
    def _cpds_array(self) -> np.ndarray:
        return self.cpds_array
    
    @property
    def _errors_array(self) -> np.ndarray:
        return self.errors_array
    

    
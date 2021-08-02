import numpy as np
import pandas as pd
import warnings

import statsmodels
from statsmodels.api import MixedLM
from scipy.stats import multivariate_normal
from scipy.special import logit, expit
from db_queries import get_covariate_estimates
from test_support.profile_support import profile
from gbd import constants as gbd

from chronos.utils import concat


class Extrapolate(object):

    @profile
    def __init__(self, df, id_col, time_col,
                 response_col, gbd_round_id, decomp_step,
                 feature_col='sdi', sdi_df=None,
                 effect_col=None, first_time=1990,
                 last_time=gbd.GBD_ROUND, back_to=1980):

        self.id_col = list(np.atleast_1d(id_col))
        self.time_col = time_col
        self.feature_col = feature_col
        self.response_col = response_col
        self.gbd_round_id = gbd_round_id
        self.decomp_step = decomp_step
        self.effect_col = effect_col
        self.first_time = first_time
        self.last_time = last_time
        self.back_to = back_to
        self.formula = self.create_formula_string()
        self.sdi_df = self.merge_SDI(df, cov=sdi_df,
                                     gbd_round_id=gbd_round_id,
                                     decomp_step=decomp_step)
        self._fit = None
        self._params = None
        self._beta = None
        self._alpha = None
        self._rand_effects_df = None
        self._cov_matrix = None
        self._param_draws = None

    @profile
    def create_formula_string(self):
        """
        Creates a formula string parseable
        by statsmodels.api.from_formula
        Currently this only works for
        linear models with 1 feature
        TODO: expand to work for
        generalized linear models
        """
        sep = ' ~ '
        return sep.join([self.response_col, self.feature_col])

    @profile
    def create_estimate_rows(self, df):
        """
        Create rows for extrapolation years based
        on the groups existing in the dataframe
        passed to the class on instantiation
        """

        estimate_df = pd.DataFrame(columns=self.id_col)
        if self.effect_col is not None:
            group_by_these = self.id_col + [self.effect_col]
        else:
            group_by_these = self.id_col

        grouped = df.groupby(group_by_these)
        years = list(range(self.back_to, self.first_time))

        for name, grp in grouped:
            temp_df = pd.DataFrame(years, columns=[self.time_col])
            for col in group_by_these:
                try:
                    temp_df[col] = grp[col].iloc[0]
                except IndexError:
                    raise ValueError(
                        "There are missing values in your id columns."
                        " Check your inputs for col {}".format(col))
            estimate_df = concat([estimate_df, temp_df], copy=False)
        return estimate_df

    @staticmethod
    @profile
    def merge_SDI(df, gbd_round_id, decomp_step, cov=None,
                  on=['location_id', 'year_id']):
        """
        Merges SDI onto a given dataframe
        by location/year
        TODO: We don't need to do this if SDI
        is not the feature variable.
        """

        for col in on:
            df[col] = df[col].astype(int)
        if cov is None:
            cov = get_covariate_estimates(
                covariate_id=881, gbd_round_id=gbd_round_id,
                decomp_step=decomp_step
            )[on + ['mean_value']]
        merged_df = df.merge(cov, on=on)
        merged_df.rename(columns={'mean_value': 'sdi'}, inplace=True)
        return merged_df

    @profile
    def fit_mixed_lm(self):
        """
        Fits a mixed linear model and stores
        the model parameters in class attributes
        """

        # fit model
        model = MixedLM.from_formula(
            self.formula, self.sdi_df[[
                self.feature_col, self.response_col, self.effect_col]],
            groups=self.sdi_df[self.effect_col])
        result = model.fit()
        self._beta = result.params[self.feature_col]
        self._alpha = result.params['Intercept']
        try:
            # statsmodels 0.8.0
            self._cov_matrix = result.cov_params().drop(
                ['groups RE']).drop(['groups RE'], axis=1)
        except KeyError:
            # statsmodels 0.9.0
            self._cov_matrix = result.cov_params().drop(
                ['Group Var']).drop(['Group Var'], axis=1)
        except ValueError:
            self._cov_matrix = result.cov_params()
        if isinstance(result.random_effects, dict):
            self._rand_effects_df = pd.DataFrame(
                result.random_effects).transpose()
            self._rand_effects_df.rename(columns={'groups': 'Intercept'},
                                         inplace=True)
        else:
            self._rand_effects_df = result.random_effects
        self._params = result.params
        self._fit = result
        return result

    @profile
    def gen_estimates(self, estimate_df=None, uncert=False):
        # create estimate rows if needed
        if estimate_df is None:
            estimate_df = self.create_estimate_rows(self.sdi_df)
        if self.feature_col not in estimate_df.columns:
            estimate_df = self.merge_SDI(estimate_df,
                                         gbd_round_id=self.gbd_round_id,
                                         decomp_step=self.decomp_step)

        # merge on random effects intercepts, if they exist
        if self._rand_effects_df is not None:
            estimate_df = estimate_df.merge(
                self._rand_effects_df,
                left_on=self.effect_col,
                right_index=True)
            estimate_df.rename(
                columns={'Intercept': 'rand_effect'}, inplace=True)
        estimate_df.reset_index(drop=True, inplace=True)

        # if we are returning draws, use parameter draws to generate them
        if uncert:
            draw_cols = [
                'draw_{}'.format(x) for x in range(len(self._param_draws))]
            alphas = np.array(self._param_draws['intercept'])
            betas = np.array(self._param_draws[self.feature_col])
            draw_matrix = np.mat(
                estimate_df[self.feature_col]).transpose() * np.mat(betas)
            draw_matrix = draw_matrix + np.mat(alphas)
            if self._rand_effects_df is not None:
                try:
                    # statsmodels 0.8.0
                    draw_matrix = draw_matrix + np.mat(
                        estimate_df['rand_effect']).transpose()
                except KeyError:
                    # statsmodels 0.9.0
                    draw_matrix = draw_matrix + np.mat(
                        estimate_df['Group']).transpose()
            draw_df = pd.DataFrame(draw_matrix, columns=draw_cols)
            estimate_df = concat([estimate_df, draw_df], axis=1)

        # otherwise return a 'mean' column
        else:
            estimate_df[self.response_col] = (
                estimate_df[self.feature_col] *
                self._beta +
                self._alpha)
            if self._rand_effects_df is not None:
                try:
                    # statsmodels 0.8.0
                    estimate_df[self.response_col] = (
                        estimate_df[self.response_col] +
                        estimate_df['rand_effect']
                    )
                except KeyError:
                    # statsmodels 0.9.0
                    estimate_df[self.response_col] = (
                        estimate_df[self.response_col] +
                        estimate_df['Group']
                    )
        return estimate_df

    @profile
    def gen_param_draws(self, draw_number=1000):
        """
        Generates a distribution of model
        parameters
        """
        if self._cov_matrix is None:
            raise ValueError("A model needs to be fit before you can generate "
                             "uncertainty, self._cov_matrix cannot be None")
        cov_cols = self._cov_matrix.columns
        mean = self._params[cov_cols]
        rm_dist = multivariate_normal(mean=mean, cov=self._cov_matrix)
        rm_array = rm_dist.rvs(size=draw_number)
        if len(cov_cols) < 3:
            rm_df = pd.DataFrame(rm_array,
                                 columns=['intercept', self.feature_col])
        else:
            try:
                # potentially only statsmodels 0.8.0
                rm_df = pd.DataFrame(
                    rm_array,
                    columns=['intercept', self.feature_col, 're_intercept'])
            except KeyError as e:
                # If a KeyError is caught, the likely culprit is API changes
                # between statsmodels 0.8.0 and 0.9.0. Raise an error and ask
                # user for more information so we can find the statsmodels API
                # inconsistencies.
                sm_version = statsmodels.__version__
                raise RuntimeError(
                    "statsmodels version {ver} detected. Current feature of "
                    "extrapolate only available in statsmodels 0.8.0. Please "
                    "submit a ticket to Central Computation with your full "
                    "function call and error traceback: {e}"
                    .format(ver=sm_version, e=e)
                )
        self._param_draws = rm_df
        return rm_df


class AdjustExtrapolate(object):

    @profile
    def __init__(self, extrapolate, estimate_df, adjust_df,
                 data_col):

        if extrapolate._fit is None:
            raise ValueError(
                "AdjustExtrapolate requires that the Extrapolation object "
                "passed has a fitted model (via Extrapolate.fit_mixed_lm"
                "for example), not None")
        self._extr = extrapolate
        self.estimate_df = estimate_df
        self.id_col = extrapolate.id_col
        self.time_col = extrapolate.time_col
        self.first_time = extrapolate.first_time
        self.effect_col = extrapolate.effect_col
        self.adjust_df = adjust_df.loc[
            adjust_df[self.time_col] == self.first_time]
        self.data_col = list(np.atleast_1d(data_col))
        self.ft_estimates_df = self.first_time_estimates()

    @profile
    def first_time_estimates(self):
        """
        Generate estimates based on a linear regression
        for the last year for which we have data.
        """

        adjust_df = self.adjust_df.copy(deep=True)

        if self._extr._param_draws is None:
            uncert = False
        else:
            uncert = True

        not_data_cols = [col for col in adjust_df if col not in self.data_col]
        ft_estimates_df = self._extr.gen_estimates(
            estimate_df=adjust_df[not_data_cols], uncert=uncert)
        ft_estimates_df.reset_index(drop=True, inplace=True)

        # reset cat columns to dtype category so that we can sort later
        cat_cols = adjust_df.loc[
            :, adjust_df.dtypes == 'category'].columns.tolist()
        for col in cat_cols:
            ft_estimates_df[col] = ft_estimates_df[col].astype('category')
        return ft_estimates_df

    @profile
    def adjust_estimates(self):
        """
        Shift the regression line by the residual
        between the last known data point and its estimate.
        """

        residuals = self.adjust_df.copy()

        # sort and reindex dfs so that they are aligned
        res_cols = [
            col for col in residuals.columns if col not in self.data_col]
        ft_est_cols = [
            col for col in self.ft_estimates_df if col not in self.data_col]
        common_cols = list(set(res_cols) & set(ft_est_cols))

        # we need to drop columns with Nan from the id columns or it will break
        # the broadcasting
        nan_cols = [
            col for col in common_cols if residuals[col].isnull().values.any()]

        common_cols = [col for col in common_cols if col not in nan_cols]
        residuals.drop(nan_cols, axis=1)
        self.ft_estimates_df.drop(nan_cols, axis=1)

        residuals.sort_values(
            by=common_cols,
            inplace=True)
        residuals.reset_index(drop=True, inplace=True)
        self.ft_estimates_df.sort_values(
            by=common_cols,
            inplace=True)
        self.ft_estimates_df.reset_index(drop=True, inplace=True)

        # subtract the estimated values from the values of the real datapoints
        residuals[self.data_col] = (
            residuals[self.data_col] -
            self.ft_estimates_df[self.data_col])

        if self.effect_col is not None:
            group_by_these = self.id_col + [self.effect_col]
        else:
            group_by_these = self.id_col

        # sort and multi-index the dfs so that we can broadcast
        # the residuals to each year we are estimating for

        self.estimate_df.sort_values(group_by_these, inplace=True)
        self.estimate_df.set_index(group_by_these, drop=False, inplace=True)
        residuals.sort_values(group_by_these, inplace=True)
        residuals.set_index(group_by_these, drop=False, inplace=True)

        self.estimate_df[self.data_col] = (
            self.estimate_df[self.data_col] + residuals[self.data_col])

        self.estimate_df.reset_index(drop=True, inplace=True)

        # add back in those stupid Nan columns
        self.estimate_df[nan_cols] = np.nan
        return self.estimate_df


@profile
def transform_data(df, data_col, direction, how='logit'):
    """Transform numbers in data_col to either log
    or logit space.
    Note: if logit transform is chosen, all numbers must
    be between 0 and 1.
    """

    data_col = list(np.atleast_1d(data_col))
    transforms = ['log', 'logit']
    if how not in transforms:
        raise ValueError("{how} is not a supported transform type, only the "
                         "following types are supported: "
                         "{transforms}".format(how=how,
                                               transforms=transforms))
    if direction not in ["forward", "back"]:
        raise ValueError("{direction} is not a supported direction type, "
                         "direction must be either 'forward' or "
                         "'back'".format(direction=direction))

    for col in data_col:
        df[col].replace(0, 1e-12, inplace=True)
    if how == 'log':
        for col in data_col:
            if direction == 'forward':
                df[col] = np.log(df[col])
            else:
                df[col] = np.exp(df[col])

    if how == 'logit':
        for col in data_col:
            df[col].replace(1, 0.999999999, inplace=True)
        for col in data_col:
            if direction == 'forward':
                df[col] = logit(df[col])
            else:
                df[col] = expit(df[col])

    return df


@profile
def extrapolate_SDI(data_df, draw_df=None, id_col=['location_id', 'sex_id'],
                    time_col='year_id', response_col='mean',
                    draw_response_col=None, feature_col='sdi',
                    sdi_df=None, effect_col='age_group_id', first_time=1990,
                    back_to=1980, transform=None,
                    gbd_round_id=gbd.GBD_ROUND_ID, decomp_step=None):
    """
    Uses linear regression to estimate data for
    dates previous to 1990.  Outputs a dataframe
    of either mean estimates or draws for the demographics
    given, for years from 'back_to' to 'first_time'

    Args:
        data_df(DataFrame): a dataframe containing mean estimates
            for the parameter being estimated, for all gbd_years.
            Note: for best results this dataframe should contain
            estimates for all most detailed locations, for all age groups.

        draw_df(DataFrame, optional): if draws will be returned, this dataframe
            should contain draws for all demographics being estimated,
            for the year 'first_time'.  Ok to contain data for more years,
            but won't effect the estimates.
            Default is None

        id_col(str or strlist, optional): a list of names of
            columns in data_df/draw_df which uniquely identify the
            demographics. Should NOT include time_col, response_col,
            feature_col, or effect_col.  Should include any additional
            column names which the original data_df/draw_df contains
            and the user wants to include in the output dataframe.
            In the case where draws are being generated this applies
            only to the draw_df. If no draw_df is passed, this argument
            applies to the data_df.
            Default is ['location_id', 'sex_id'].

        time_col(str, optional): name of the column in data_df
            and draw_df which contains years.
            Default is 'year_id'

        response_col(str, optional): name of the column in data_df
            that contains the mean estimates for the parameter
            to be extrapolated.
            Default is 'mean'

        draw_response_col(str or strlist): list of the 'draw' columns in
            draw_df.  Typically [draw_{}.format(x) for x in range(1000)]
            Default is None

        feature_col(str, optional): name of the variable which
            will be used to create a link between existing estimates and
            the year.  Currently only 'sdi' is supported.
            Default is 'sdi'

        sdi_df(df, optional): dataframe of sdi values. If specified, will
            save memory/db load by using the passed in dataframe instead of
            calling get_covariate_estimates() inside

        effect_col(str, optional): name of the column containing
            ids for random effects on the model.  Each 'group' in
            this column will recieve unique estimates based on their
            estimated effect on the model.  Typically, age groups.
            Default is 'age_group_id'

        first_time(int, optional): the earliest year for which estimates
            exist.
            Default is 1990

        back_to(int, optional): the earliest year to create new estimates for.
            All years between 'back_to' and 'first_time' will be estimated for.
            Default is 1980

        transform(str, optional): The transform to be applied to the data
            before the linear regression takes place. Currently only 'log'
            and 'logit' are supported.
            NOTE: data should be transformed using a method which is
            consistent with the measure type being estimated.
            Default is None

        gbd_round_id(int, optional): Defaults to current round. Used
            to determine which version of SDI covariates to retrieve.
            If sdi_df is not None, this argument is not used.

    Returns:
        Dataframe

    Examples:
        To create estimate draws for a cause using SDI:
            df = extrapolate(df, draw_df=my_draw_df,
            id_col=['location_id', 'sex_id', 'cause_id', 'model_version_id'],
            response_col='mean_prevalence',
            draw_response_col=[draw_{}.format(x) for x in range(1000)])

        To create estimates for a covariate using SDI:
            df = extrapolate(df,
            id_col=['location_id', 'sex_id', 'covariate_id']
            response_col='mean')

    """

    # transform the data in whichever way is appropriate for the measure
    draw_response_col = list(np.atleast_1d(draw_response_col))
    if draw_df is not None:
        if draw_df[draw_response_col].isnull().values.any():
            draw_df[draw_response_col].dropna(inplace=True)
            warnings.warn((
                "Null values detected in dataframe: "
                "all rows containing null values have been dropped",
                RuntimeWarning))
    if transform is not None:
        data_df = transform_data(
            data_df, response_col, direction='forward', how=transform)

    ext = Extrapolate(data_df, id_col, time_col, response_col,
                      gbd_round_id=gbd_round_id, decomp_step=decomp_step,
                      feature_col=feature_col, sdi_df=sdi_df,
                      effect_col=effect_col, first_time=first_time,
                      back_to=back_to)

    # If extrapolate is being used to extrapolate draws, a draw_df
    # with 1990s draws is required
    if draw_df is not None:
        if not draw_response_col:
            raise ValueError("draw_response_col is None, If draw_df is "
                             "provided, a draw_response_col must also be "
                             "provided")
        estimate_df = ext.create_estimate_rows(draw_df)
        if transform is not None:
            draw_df = transform_data(
                draw_df, draw_response_col, direction='forward', how=transform)
        uncert = True
    else:
        estimate_df = None
        draw_df = data_df.loc[data_df[time_col] == first_time].copy()
        draw_response_col = response_col
        uncert = False

    ext.fit_mixed_lm()
    if uncert:
        ext.gen_param_draws(draw_number=len(draw_response_col))
    estimate_df = ext.gen_estimates(estimate_df=estimate_df, uncert=uncert)
    adj = AdjustExtrapolate(ext, estimate_df, draw_df, draw_response_col)
    adj_estimate_df = adj.adjust_estimates()

    if transform is not None:
        adj_estimate_df = transform_data(
            adj_estimate_df, draw_response_col,
            direction='back', how=transform)

    # drop some columns the user probably doesn't want/need --
    # rand_effect is from the statsmodels 0.8.0, Group is from version 0.9.0.
    adj_estimate_df.drop(
        ['rand_effect', 'Group', 'sdi'], axis=1, inplace=True, errors='ignore'
    )

    return adj_estimate_df
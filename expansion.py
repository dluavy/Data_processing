import warnings
import pandas as pd
import numpy as np
import logging

from core_maths.interpolate import pchip_interpolate, linear_interpolate
from draw_sources.draw_sources import DrawSource
from gbd import constants as gbd
from test_support.profile_support import profile

from chronos.extrapolate_core import extrapolate_SDI, Extrapolate
from chronos.utils import concat


logger = logging.getLogger(__name__)

VALID_EXTRAPOLATION_MEASURES = {18, 5, 6, 15, 19}
VALID_EXTRAPOLATION_SOURCES = [
    'epi',
    'exposure'
]


@profile
def interpolate(
        input_data,
        gbd_id,
        gbd_id_type='cause_id',
        measure_id=None,
        location_id=None,
        age_group_id=None,
        sex_id=None,
        year_start_id=None,
        year_end_id=None,
        interp_method='pchip',
        draw_prefix='draw_'
):
    """
    Core chronos function for interpolating draws. Operates on an in-memory
    pandas dataframe or a DrawSource. Returns dataframe with interpolated
    draws.

    Args:
     input_data (dataframe or Custom DrawSource): Either a pandas dataframe
         orr DrawSource containing input draws to interpolate.
     year_start_id (Optional[int]): minimum value for year_id to include in
        the output dataframe. The output dataframe will contain rows
        for all year_ids between year_start_id and year_end_id.  The actual
        interpolation algorithm will utilize all year_ids in the input_data
        dataframe to create the best estimates.
        The default (None), uses the minimum value in the year_id column.
     year_end_id (Optional[int]): maximum value for year_id to include in
        the output dataframe.  See description for year_start_id.
        The default (None), uses the maximum value in the year_id column.
     gbd_id (intlist): list of gbd ids (ie list of cause_ids)
     gbd_id_type (str): type of gbd_ids being requested. default 'cause_ids'
     gbd_round_id (int): gbd round id of data being requested. default
         current gbd round id
     measure_id (intlist): measures to interpolate for. default
         None, which will operate on all measures in the df
     location_id (intlist): locations to be queried.  default
         None, which will operate on all locations in the df
     age_group_id (intlist): ages to be queried. default
         None, which will operate on all ages in the df
     sex_id (intlist): sex to be queried. default
         None, which will operate on all sexes in the df
     interp_method (str): Interpolation method to use. Right now,
         'pchip' for a cubic spline interpolation, or 'linear'
         for linear interpolation are the options available
     draw_prefix (str): Prefix on draws in data, defaults to 'draw_'

     Returns:
         Dataframe:
             location_id, year_id, age_group_id, sex_id, measure_id,
             cause_id/modelable_entity_id, model_version_id, draw_0-draw_999

     Raises:
         ValueError: If input_data does not contain year_id, measure_id,
             location_id, age_group_id, and sex_id
    """
    if not isinstance(input_data, pd.DataFrame):
        if not isinstance(input_data, DrawSource):
            raise ValueError('input_data arg must be a pandas Dataframe or a '
                             'DrawSource object, type() of input given is: '
                             '{type_input}'.format(type_input=type(input_data)))
        arg_dict = {gbd_id_type: gbd_id, 'measure_id': measure_id,
                    'location_id': location_id, 'age_group_id': age_group_id,
                    'sex_id': sex_id}
        filters = {k: v for k, v in arg_dict.items() if v is not None}
        df = input_data.content(filters=filters)
        if 'model_version_id' in df.columns:
            raise ValueError('model_version_id should not be in columns, '
                             'data frame columns passed were: '
                             '{columns}'.format(columns=df.columns))
    else:
        df = input_data.copy(deep=True)

    base_cols = ['year_id', 'measure_id', 'location_id', 'age_group_id',
                 'sex_id', 'metric_id']
    for column in base_cols:
        if column not in df.columns:
            raise ValueError("{} column is missing from df".format(column))

    draw_cols = [col for col in df.columns if col.startswith(draw_prefix)]
    if len(draw_cols) == 0:
        raise ValueError("There are no draw columns (draw_) in the data frame, "
                         "draw columns must be present in order for "
                         "interpolation")
    df.drop(['pop', 'population', 'env', 'envelope'],
            axis=1, inplace=True, errors='ignore')
    time_col = 'year_id'
    # we have to cast the year_id column to integers for pchip interpolate
    try:
        df[time_col] = df[time_col].astype(int)
    except ValueError:
        raise("The {} column of the input dataframe cannot be cast to "
              "integers. Currently we can only interpolate across an "
              "integer time series".format(time_col))

    id_cols = list(set(df.columns) - set(draw_cols) - set([time_col]))
    if ['model_version_id'] in id_cols:
        raise ValueError('model_version_id should not be in id_columns, '
                         'id_columns passed were: '
                         '{columns}'.format(columns=id_cols))

    if gbd_id is not None:
        try:
            df = df.loc[df[gbd_id_type].isin(np.atleast_1d(gbd_id))]
        except KeyError:
            logger.warning("Draws are missing {} column, interpolate will run "
                           "on all gbd_ids present".format(gbd_id_type))

    if measure_id is not None:
        measure_id = list(np.atleast_1d(measure_id))
        df_measures = df['measure_id'].unique().tolist()
        if [m for m in measure_id if m in df_measures]:
            df = df.loc[df['measure_id'].isin(measure_id)]
        else:
            raise ValueError("None of measure_id {} found in input "
                             "dataframe.".format(measure_id))
    if location_id is not None:
        location_id = list(np.atleast_1d(location_id))
        df_locs = df['location_id'].unique().tolist()
        if [l for l in location_id if l in df_locs]:
            df = df.loc[df['location_id'].isin(location_id)]
        else:
            raise ValueError("None of location_id {} found in input "
                             "dataframe.".format(location_id))
    if age_group_id is not None:
        age_group_id = list(np.atleast_1d(age_group_id))
        df_ages = df['age_group_id'].unique().tolist()
        if [a for a in age_group_id if a in df_ages]:
            df = df.loc[df['age_group_id'].isin(age_group_id)]
        else:
            raise ValueError("None of age_group_id {} found in input "
                             "dataframe.".format(age_group_id))
    if sex_id is not None:
        sex_id = list(np.atleast_1d(sex_id))
        df_sexes = df['sex_id'].unique().tolist()
        if [s for s in sex_id if s in df_sexes]:
            df = df.loc[df['sex_id'].isin(sex_id)]
        else:
            raise ValueError("None of sex_id {} found in input "
                             "dataframe.".format(sex_id))

    if interp_method == 'pchip':
        interp = pchip_interpolate(df, id_cols, draw_cols)
    elif interp_method == 'linear':
        interp = linear_interpolate(df, id_cols, draw_cols)
    else:
        raise NotImplementedError(
            '{} is not a supported interpolation method.'.format(
                interp_method))
    if year_start_id is not None and year_end_id is not None:
        if year_start_id >= year_end_id:
            raise ValueError("year_start_id must be stricly less "
                             "than year_end_id. Given: "
                             "year_start_id = {}, "
                             "year_end_id = {}".format(
                year_start_id, year_end_id))
        df_years = df['year_id'].unique().tolist()
        if year_start_id < min(df_years):
            raise ValueError("Invalid entry for year_start_id. Base "
                             "interpolate does not extrapolate. Valid entries "
                             "must be greater than or equal to minimum year "
                             "in data, {}.".format(min(df_years)))
        if year_end_id > max(df_years):
            raise ValueError("Invalid entry for year_end_id. Base "
                             "interpolate does not extrapolate. Valid entries "
                             "must be less than or equal to maximum year "
                             "in data, {}.".format(max(df_years)))
    if year_start_id is not None:
        df = df.loc[df['year_id'] >= year_start_id]
        interp = interp.loc[interp['year_id'] >= year_start_id]
    if year_end_id is not None:
        df = df.loc[df['year_id'] <= year_end_id]
        interp = interp.loc[interp['year_id'] <= year_end_id]

    return concat([df, interp])


@profile
def extrapolate(mean_df,
                draw_df,
                source,
                gbd_id=None,
                gbd_id_type='cause_id',
                measure_id=None,
                location_id=None,
                age_group_id=None,
                sex_id=None,
                gbd_round_id=gbd.GBD_ROUND_ID,
                decomp_step=None):
    """
    Returns extrapolated df. Will either copy 90's or use linear
    regression, depending on if draws are from dismod/exposure and
    are the correct measure (see note below).

    Args:
     mean_df (dataframe): dataframe with _id cols and 'mean'. Only used
         if back-extrapolation via linear regression will occur.
         Otherwise None.
     draw_df (dataframe): draw df that must contain data for 1990
     source: where the draws come from, i.e. epi, como, codcorrect
     gbd_id (intlist): List of ids in your dataframe (ie list of cause_ids)
     gbd_id_type (str): type of gbd_ids being requested. default 'cause_id'
     measure_id (intlist): measures to interpolate for. default
         None, which will operate on all measures in the df
     location_id (intlist): locations to be queried.  default
         None, which will operate on all locations in the df
     age_group_id (intlist): ages to be queried. default
         None, which will operate on all ages in the df
     sex_id (intlist): sex to be queried. default
         None, which will operate on all sexes in the df
     gbd_round_id (int): Which gbd round is this data from? Used
         to determine which version of SDI covariate to use
         during extrapolation

     Returns:
         Dataframe:
             location_id, year_id, age_group_id, sex_id, measure_id,
             cause_id/modelable_entity_id, model_version_id, draw_0-draw_999

     Raises:
         ValueError: If df does not contain year_id, measure_id,
             location_id, age_group_id, and sex_id;
             or if draw_df is missing 1990
     Note: measures 5, 6, 8, 15, and 19 coming out of dismod or risk
           will be back-extrapolated using linear regression.
           All other measures and sources will just have 1990 copied and
           added as the 1980's. For questions on this methodology talk to
           Caitlyn Steiner or Kyle Foreman
    """
    for column in ['year_id', 'measure_id', 'location_id', 'age_group_id',
                   'sex_id', 'metric_id']:
        if column not in draw_df.columns:
            raise ValueError("The column {} is missing from draw_df and is "
                             "required for extrapolation".format(column))
    if 1990 not in draw_df['year_id'].unique():
        raise ValueError("draw_df is missing the year 1990, which is required "
                         "for extrapolation")
    draw_df = draw_df.loc[draw_df['year_id'] == 1990]

    if mean_df is not None:
        valid_measure = [m for m in VALID_EXTRAPOLATION_MEASURES
                         if m in set(mean_df.measure_id.tolist())]
        valid_source = source in VALID_EXTRAPOLATION_SOURCES
        if not(valid_source and valid_measure):
            raise RuntimeError(
                "DF of means was supplied but the source is not dismod/epi, "
                "or valid measures were not supplied. Given source "
                "{source} and measures {measures}".format(
                                source=valid_source,
                                measures=mean_df.measure_id.unique().tolist()))

    if gbd_id is not None:
        try:
            gbd_id = np.atleast_1d(gbd_id).tolist()
            draw_df = draw_df.loc[draw_df[gbd_id_type].isin(gbd_id)]
        except KeyError:
            logger.warning(
                ("Draws are missing {} column, extrapolate will run on all "
                 "gbd_ids present".format(gbd_id_type)))

    if measure_id is not None:
        measure_id = np.atleast_1d(measure_id).tolist()
        draw_df = draw_df.loc[draw_df['measure_id'].isin(measure_id)]
    else:
        measure_id = list(draw_df.measure_id.unique())

    if location_id is not None:
        location_id = np.atleast_1d(location_id).tolist()
        draw_df = draw_df.loc[draw_df['location_id'].isin(location_id)]
        location_id = list(draw_df.location_id.unique())

    if age_group_id is not None:
        age_group_id = np.atleast_1d(age_group_id).tolist()
        draw_df = draw_df.loc[draw_df['age_group_id'].isin(age_group_id)]
        age_group_id = list(draw_df.age_group_id.unique())

    if sex_id is not None:
        sex_id = np.atleast_1d(sex_id).tolist()
        draw_df = draw_df.loc[draw_df['sex_id'].isin(sex_id)]
        sex_id = list(draw_df.sex_id.unique())

    transform_dict = {18: 'logit', 5: 'logit', 6: 'log', 15: 'logit',
                      11: 'log', 19: 'log'}
    extrap_measures = VALID_EXTRAPOLATION_MEASURES.copy()
    remove_measures = []
    for m in extrap_measures:
        if m not in draw_df['measure_id'].unique():
            remove_measures.append(m)
    # if source is epi/risk we may need to do extrapolation
    # depending on measures. Otherwise copy from 90
    if source in VALID_EXTRAPOLATION_SOURCES:
        extrap_measures = extrap_measures - set(remove_measures)
        copy_measures = list(set(measure_id) - extrap_measures)
    else:
        extrap_measures = []
        copy_measures = measure_id

    df_list = []
    if mean_df is not None and extrap_measures:
        if not all(m in mean_df.measure_id.unique() for m in extrap_measures):
            raise ValueError("Measure id {m} is missing in mean_df".format(m=m))
        draw_df = Extrapolate.merge_SDI(
            draw_df, gbd_round_id=gbd_round_id, decomp_step=decomp_step)
        draw_cols = [col for col in draw_df.columns if 'draw_' in col]
        for m in extrap_measures:
            transform_type = transform_dict[m]
            df_list.append(extrapolate_SDI(
                data_df=mean_df, draw_df=draw_df,
                id_col=[col for col in draw_df.columns if not
                        col.startswith('draw_') and col not in
                        ['year_id', 'age_group_id', 'sdi']],
                time_col='year_id', response_col='mean',
                draw_response_col=draw_cols, feature_col='sdi',
                effect_col='age_group_id', first_time=1990,
                back_to=1980, transform=transform_type,
                gbd_round_id=gbd_round_id, decomp_step=decomp_step))
    if copy_measures:
        warnings.warn('Extrapolate back to 1980 not allowed for measure {}, '
                      'source {}. Copying 1990 draws for '
                      'the 80s.'.format(measure_id, source))
        ninety = draw_df.loc[draw_df['year_id'] == 1990]
        ninety = ninety.loc[ninety['measure_id'].isin(copy_measures)]
        if len(ninety) == 0:
            raise RuntimeError("Data frame for year 1990 is empty!")
        for year in range(1980, 1990):
            eighty = ninety.copy(deep=True)
            eighty['year_id'] = year
            df_list.append(eighty)
    return concat(df_list).reset_index(drop=True)
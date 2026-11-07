// APPEND to the already working GEE_ES_2018_coverage script.
// Requires points, image, valid, kernel, climate, eraProjection, PROJECTION, RASTER.
// Four NEW tasks; do not rerun the older coverage export task.
// Exclusions were fixed from the audited coverage CSV, before any model fit.
var EXCLUDED_IDS = [
  '27742384','27922352','28562440','28981598','31462364',
  '32382024','32462348','32722342','33581694'
];
var patchPoints = points
  .filter(ee.Filter.inList('point_id',EXCLUDED_IDS).not())
  .sort('point_id');
print('PATCH SOURCE COUNT — expected 3754',patchPoints.size());
print('PATCH UNIQUE IDS — expected 3754',
  patchPoints.aggregate_count_distinct('point_id'));

// Zero fill only makes the array image exportable everywhere.
// Every retained patch must still have valid=1 at all 289 pixels on validation.
var arrayImage = image.unmask(0,false)
  .addBands(valid.rename('valid'))
  .neighborhoodToArray(kernel);
var PATCH_PROPERTIES = [
  'point_id','POINTID','NUTS_0','OC','Depth','SURVEY_DATE',
  'LC','LC0_Desc','LC1_Desc','source_lon','source_lat',
  'source_x3035','source_y3035',
  'tp_winter_m','t2m_summer_C','sm_annual','era_grid_x','era_grid_y'
];

for (var batch=0;batch<4;batch++) {
  var subset = ee.FeatureCollection(patchPoints.toList(1000,batch*1000));
  var withClimate = climate.reduceRegions({
    collection:subset,reducer:ee.Reducer.first(),
    crs:eraProjection,tileScale:4
  });
  var patches = arrayImage.sampleRegions({
    collection:withClimate,properties:PATCH_PROPERTIES,
    projection:PROJECTION,scale:20,geometries:true,tileScale:4
  }).map(function(f) {
    var xy=f.geometry().transform('EPSG:3035',0.1).coordinates();
    var ll=f.geometry().transform('EPSG:4326',0.1).coordinates();
    return f.set({predictor_year:2018,
      patch_center_x3035:xy.get(0),patch_center_y3035:xy.get(1),
      patch_center_lon:ll.get(0),patch_center_lat:ll.get(1)});
  });
  var taskName='SOC_ES_2018_patch17_part'+(batch+1);
  Export.table.toDrive({
    collection:patches,description:taskName,
    folder:'SOC_Spain_replication',fileNamePrefix:taskName,fileFormat:'CSV',
    selectors:PATCH_PROPERTIES.concat([
      'predictor_year','patch_center_x3035','patch_center_y3035',
      'patch_center_lon','patch_center_lat'
    ]).concat(RASTER).concat(['valid'])
  });
}
print('Run ONLY patch17_part1, part2, part3, part4 tasks.');
print('Expected rows per part: 1000, 1000, 1000, 754.');

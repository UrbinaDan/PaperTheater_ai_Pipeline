import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely import make_valid

def mask_to_contours(mask):
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_to_polygon(contour, simplify_tolerance=2.0):
    pts = contour.squeeze()
    if len(getattr(pts, "shape", [])) != 2 or len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    poly = poly.simplify(simplify_tolerance, preserve_topology=True)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly

def mask_to_polygons(mask, simplify_tolerance=2.0):
    polys = []
    contours = mask_to_contours(mask)
    for c in contours:
        poly = contour_to_polygon(c, simplify_tolerance)
        if poly is None:
            continue
        if isinstance(poly, Polygon):
            polys.append(poly)
        elif isinstance(poly, MultiPolygon):
            polys.extend(list(poly.geoms))
    return polys
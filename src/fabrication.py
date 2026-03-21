from shapely.ops import unary_union
from shapely import make_valid

def merge_polygons(polys):
    if not polys:
        return None
    geom = unary_union(polys)
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom

def thicken_fragile_parts(geom, amount=2):
    if geom is None:
        return None
    geom = geom.buffer(amount).buffer(-amount)
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom

def remove_tiny_parts(geom, min_area=50):
    if geom is None:
        return None

    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_area else None

    if geom.geom_type == "MultiPolygon":
        kept = [g for g in geom.geoms if g.area >= min_area]
        if not kept:
            return None
        merged = unary_union(kept)
        if not merged.is_valid:
            merged = make_valid(merged)
        return merged

    return None
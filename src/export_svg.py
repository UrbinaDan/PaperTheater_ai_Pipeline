import svgwrite

def polygon_path(poly):
    coords = list(poly.exterior.coords)
    d = f"M {coords[0][0]},{coords[0][1]} " + " ".join(f"L {x},{y}" for x, y in coords[1:]) + " Z"
    return d

def save_svg(geom, out_path, width, height):
    dwg = svgwrite.Drawing(str(out_path), size=(width, height))
    if geom is None:
        dwg.save()
        return

    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)

    for poly in geoms:
        dwg.add(dwg.path(
            d=polygon_path(poly),
            fill="none",
            stroke="black",
            stroke_width=1
        ))

    dwg.save()
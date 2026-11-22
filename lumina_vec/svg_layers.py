"""Reference-style boundary underlay followed by line-constrained color fills and dedicated line-art layer."""
import copy
import xml.etree.ElementTree as ET


def boundary_then_fill(svg, stroke_width=2, *, hidden_strokes=(), metadata=None, line_region_ids=None):
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    root = ET.fromstring(svg)
    ns = '{http://www.w3.org/2000/svg}'
    if any(e.tag != ns + 'path' for e in root):
        raise ValueError('Expected generated flat filled paths')
    outlines = ET.Element(ns + 'g', {'id': 'region-boundaries'})
    fills = ET.Element(ns + 'g', {'id': 'region-colours', 'data-role': 'line-constrained-color-layer'})
    lines = ET.Element(ns + 'g', {'id': 'line-art-layer', 'data-role': 'cnn-line-art-layer'}) if line_region_ids is not None else None
    if metadata:
        outlines.attrib.update(metadata)
    line_set = set(line_region_ids) if line_region_ids is not None else set()
    outlines_line = []
    outlines_fill = []
    for index, path in enumerate(list(root)):
        color = path.get('fill')
        if not color or color == 'none':
            raise ValueError('Expected solid region fill')
        edge = copy.deepcopy(path)
        edge.set('id', f'boundary-{index}')
        edge.set('class', 'region-gap-filler')
        edge.set('fill', 'none')
        edge.set('stroke', color)
        edge.set('stroke-width', str(stroke_width))
        edge.set('stroke-linejoin', 'round')
        edge.set('stroke-linecap', 'round')
        edge.set('vector-effect', 'non-scaling-stroke')
        if color in hidden_strokes:
            edge.set('stroke-opacity', '0')
        root.remove(path)
        if lines is not None and index in line_set:
            outlines_line.append(edge)
            lines.append(path)
        else:
            outlines_fill.append(edge)
            fills.append(path)
    outlines.extend(outlines_line + outlines_fill)
    if lines is not None and len(lines) > 0:
        # Planar partition regions are mutually disjoint: emit line-art first, then color fills (matching Teacher SVGs)
        root.extend([outlines, lines, fills])
    else:
        root.extend([outlines, fills])
    return ET.tostring(root, encoding='unicode')

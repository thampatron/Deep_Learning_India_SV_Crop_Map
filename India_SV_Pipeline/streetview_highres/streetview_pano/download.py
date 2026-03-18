import itertools
import time
import concurrent.futures
from dataclasses import dataclass
from io import BytesIO
from typing import Generator, Tuple

import requests
from PIL import Image


@dataclass
class TileInfo:
    x: int
    y: int
    fileurl: str


@dataclass
class Tile:
    x: int
    y: int
    image: Image.Image


def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    """
    Returns the width and height of a panorama at a given zoom level, depends on the
    zoom level.
    """
    return 2**zoom, 2 ** (zoom - 1)


def make_download_url(pano_id: str, zoom: int, x: int, y: int) -> str:
    """
    Returns the URL to download a tile.
    """
    return (
        "https://cbk0.google.com/cbk"
        f"?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}&w=640&h=640"
    )

def make_download_url_free(pano_id: str, zoom: int, x: int, y: int) -> str:
    """
    Returns the URL to download a tile.
    """
    return (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    f"?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
    )
    # return (
    # "https://streetviewpixels-pa.googleapis.com/v1/tile"
    # f"?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
    # )


def fetch_panorama_tile(tile_info: TileInfo) -> Image.Image:
    """
    Tries to download a tile, returns a PIL Image.
    """
    while True:
        try:
            response = requests.get(tile_info.fileurl, stream=True)
            break
        except requests.ConnectionError:
            print("Connection error. Trying again in 2 seconds.")
            time.sleep(2)

    return Image.open(BytesIO(response.content))


def iter_tile_info(pano_id: str, zoom: int) -> Generator[TileInfo, None, None]:
    """
    Generate a list of a panorama's tiles and their position.
    """
    width, height = get_width_and_height_from_zoom(zoom)
    for x, y in itertools.product(range(width), range(height)):
        yield TileInfo(
            x=x,
            y=y,
            fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
        )

from math import ceil

# def iter_tilesnew(pano_id: str, zoom: int, heading: int, multi_threaded: bool = False) -> Generator[Tile, None, None]:
#     width, height = get_width_and_height_from_zoom(zoom)
#     # Calculate the number of tiles per 90 degree segment
#     tiles_per_90_deg = width // 4
#     # Calculate start and end tile based on heading
#     start_tile_x = (heading // 90) * tiles_per_90_deg - tiles_per_90_deg
#     end_tile_x = start_tile_x + tiles_per_90_deg
#     start_tile_x = start_tile_x % width
#     end_tile_x = end_tile_x % width
#
#
#    # Generate tile information taking wrapping into account
#     tile_infos = []
#     if start_tile_x < end_tile_x:
#         for y in range(height):
#             for x in range(start_tile_x, end_tile_x):
#                 tile_infos.append(TileInfo(x=x, y=y, fileurl=make_download_url(pano_id, zoom, x, y)))
#     else:
#         # Handle wrap-around
#         for y in range(height):
#             for x in range(start_tile_x, width):
#                 tile_infos.append(TileInfo(x=x, y=y, fileurl=make_download_url(pano_id, zoom, x, y)))
#             for x in range(0, end_tile_x):
#                 tile_infos.append(TileInfo(x=x, y=y, fileurl=make_download_url(pano_id, zoom, x, y)))
#
#     if not multi_threaded:
#         for info in tile_infos:
#             image = fetch_panorama_tile(info)
#             yield Tile(x=info.x, y=info.y, image=image)
#         return
#
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_tile = {
#             executor.submit(fetch_panorama_tile, info): info
#             for info in tile_infos
#         }
#         for future in concurrent.futures.as_completed(future_to_tile):
#             info = future_to_tile[future]
#             try:
#                 image = future.result()
#             except Exception as exc:
#                 print(f"{info.fileurl} generated an exception: {exc}")
#             else:
#                 yield Tile(x=info.x, y=info.y, image=image)
#
#
#
# def get_panoramanew(pano_id: str, zoom: int = 5, heading: int = 275, multi_threaded: bool = False) -> Image.Image:
#     """
#     Downloads a streetview panorama, focusing on a 90-degree view based on the specified heading.
#     """
#     tile_width = 512
#     tile_height = 512
#
#     _, total_height = get_width_and_height_from_zoom(zoom)
#     panorama_width = tile_width * (2 ** zoom) // 4  # Only one-fourth the total width
#     panorama = Image.new("RGB", (panorama_width, total_height * tile_height))
#
#     for tile in iter_tiles(pano_id=pano_id, zoom=zoom, heading=heading, multi_threaded=multi_threaded):
#         # Adjust tile placement in the panorama
#         adjusted_x = (tile.x * tile_width) - ((heading // 90) * tile_width * (2 ** zoom // 4))
#         panorama.paste(im=tile.image, box=(adjusted_x, tile.y * tile_height))
#         del tile
#
#     return panorama

def iter_tiles(
    pano_id: str, zoom: int, multi_threaded: bool = False
) -> Generator[Tile, None, None]:
    if not multi_threaded:
        for info in iter_tile_info(pano_id, zoom):
            image = fetch_panorama_tile(info)
            yield Tile(x=info.x, y=info.y, image=image)
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tile = {
            executor.submit(fetch_panorama_tile, info): info
            for info in iter_tile_info(pano_id, zoom)
        }
        for future in concurrent.futures.as_completed(future_to_tile):
            info = future_to_tile[future]
            try:
                image = future.result()
            except Exception as exc:
                print(f"{info.fileurl} generated an exception: {exc}")
            else:
                yield Tile(x=info.x, y=info.y, image=image)


def get_panorama(
    pano_id: str, zoom: int = 5, multi_threaded: bool = False
) -> Image.Image:
    """
    Downloads a streetview panorama.
    Multi-threaded is a lot faster, but it's also a lot more likely to get you banned.
    """

    tile_width = 512
    tile_height = 512

    total_width, total_height = get_width_and_height_from_zoom(zoom)
    panorama = Image.new("RGB", (total_width * tile_width, total_height * tile_height))

    for tile in iter_tiles(pano_id=pano_id, zoom=zoom, multi_threaded=multi_threaded):
        panorama.paste(im=tile.image, box=(tile.x * tile_width, tile.y * tile_height))
        del tile

    return panorama

# def iter_tile_info_side(pano_id: str, zoom: int, side: str) -> Generator[TileInfo, None, None]:
#     """
#     Generate a list of a panorama's tiles for a specific side (left or right) and their position.
#     """
#     width, height = get_width_and_height_from_zoom(zoom)
#     mid_x = width // 2  # Midpoint of the x-axis
#     tile_range_x = range(0, mid_x) if side == 'left' else range(mid_x, width)
#
#     for x, y in itertools.product(tile_range_x, range(height)):
#         yield TileInfo(
#             x=x,
#             y=y,
#             fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
#         )

def iter_tile_info_side(pano_id: str, zoom: int, side: str) -> Generator[TileInfo, None, None]:
    """
    Generate a list of a panorama's tiles for a specific side (left or right) and their position,
    focusing only on the middle 60% of that side.
    """
    width, height = get_width_and_height_from_zoom(zoom)
    mid_x = width // 2  # Midpoint of the x-axis
    if side == 'left':
        start_x = int(mid_x * 0.2)  # Start from 20% of the full width
        end_x = int(mid_x * 0.8)    # End at 80% of the full width
    else:
        start_x = mid_x - int(mid_x)  # Start from middle plus 20% of half width
        end_x = mid_x + int(mid_x * 0.6)    # End at middle plus 80% of half width

    for x, y in itertools.product(range(start_x, end_x), range(height)):
        yield TileInfo(
            x=x,
            y=y,
            fileurl=make_download_url_free(pano_id=pano_id, zoom=zoom, x=x, y=y),
        )



def get_panorama_side(pano_id: str, zoom: int, side: str, multi_threaded: bool = False) -> Image.Image:
    """
    Downloads a specific portion (middle 60%) of one side of a streetview panorama (either left or right).
    """
    tile_width = 512
    tile_height = 512

    total_width, total_height = get_width_and_height_from_zoom(zoom)
    half_width = total_width // 2
    partial_width = int(half_width * 0.6)  # Only using 60% of each half

    panorama = Image.new("RGB", (partial_width * tile_width, total_height * tile_height))

    for tile in iter_tiles_side(pano_id, zoom, side, multi_threaded):
        adjusted_x = (tile.x - (int(half_width * 0.2) if side == 'left' else (half_width + int(half_width * 0)))) * tile_width
        panorama.paste(im=tile.image, box=(adjusted_x, tile.y * tile_height))
        del tile

    # Crop the top and bottom 20% of the image
    crop_height = int(panorama.size[1] * 0.2)
    cropped_panorama = panorama.crop((0, crop_height, panorama.size[0], panorama.size[1] - 2*crop_height))

    # Crop the furthest right 10% of the image
    crop_width = int(cropped_panorama.size[0] * 0.8)
    cropped_panorama = cropped_panorama.crop((0, 0, crop_width, cropped_panorama.size[1]))

    return cropped_panorama



def iter_tiles_side(pano_id: str, zoom: int, side: str, multi_threaded: bool = False) -> Generator[Tile, None, None]:
    """
    Adjusted iter_tiles to use the side-specific iter_tile_info.
    """
    if not multi_threaded:
        for info in iter_tile_info_side(pano_id, zoom, side):
            image = fetch_panorama_tile(info)
            yield Tile(x=info.x, y=info.y, image=image)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_tile = {
                executor.submit(fetch_panorama_tile, info): info
                for info in iter_tile_info_side(pano_id, zoom, side)
            }
            for future in concurrent.futures.as_completed(future_to_tile):
                info = future_to_tile[future]
                try:
                    image = future.result()
                except Exception as exc:
                    print(f"{info.fileurl} generated an exception: {exc}")
                else:
                    yield Tile(x=info.x, y=info.y, image=image)

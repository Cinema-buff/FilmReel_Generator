# SPDX-License-Identifier: GPL-3.0-or-later
bl_info = {
    "name": "Film Reel",
    "author": "Claude",
    "version": (1, 13, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Film Reel",
    "description": "Build a 35mm film strip / reel from an image sequence or video, "
                   "using the embedded Film35mm model (M_FilmHolder + M_FilmImage materials)",
    "category": "Object",
}

import base64
import bisect
import json
import math
import os
import re
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup

# ----------------------------------------------------------------------------
# Embedded Film35mm mesh (extracted from Film35mm.fbx; zlib + base64).
# Layout: header (nverts, npolys, nloops) | verts f32*3 | poly sizes u16 |
#         loop vertex indices u16 | material index u8 per poly | UV f32*2 per loop
# Local space: X = film width, Y = frame pitch (4 perfs), Z = face normal.
# Material index 0 = M_FilmHolder (base + perforations), 1 = M_FilmImage (frame).
# ----------------------------------------------------------------------------
_MESH_B64 = (
    "eNrsuHdcFNnzLny6e1TMmOMqxtV1V9E1wDTdzRlzFsSsgDkLiGsOmDArxkVRwZzDGnEGkR6ziOIKkjG7rhHjGgj3mf1OOb7v"
    "e997/7jhr59+eLqqpk6fPqdOV1U/ywyMuYiM3SvK2MZH8fyE11bOXBKM2ZAXrA+HrBuXt0rjfb1hd/1BvgX5pw2w96gkL30c"
    "zxPLLOMsapcx+l08X/Nwk86c+1seQs7vvFVnYYMsB9ql8VcPYPevY7kDub/NnlPfMtjMTB18cE+3bfJ8yNXDcU+3rfLT286m"
    "O71g7/BQLp7kbNq/Efa+92RfyB8rYK5RaXKOzky593BP9rRVYSszjeqIe7LU01MynE0T7sKeU0heCTm9PeyBopwS8YJ36Il7"
    "Jo6Tq29+wav+jnvmjJAvQx5WDvd07id/2P+C59/H2Lg9ZpcDL/iYThgbddh8EfPe2D+Js9REud+TeL5wO2TP+UbXZGfTG8Gb"
    "s0+X5V53nE1V3z7xZIlH5PFbX/BbuV7Yq5by4agXPOsvm72Q3OSveL73M+xxmnHq3/F86X3YGTsdh7EDd2z3ZM575PbP4vmV"
    "NZBDVp1pHstMIVJjmQW6yPVTnE1O0z092eKtshnyKN1PZp5r5BZaGk/ccdfI2Kbonttf8CXD4RN1yThnxwte+WJJmUWtN1Z4"
    "Hs//6gt7XHEzfxHP42acgv8Ec+dUZ1NiIWMc6xsqH4P88lNkHAubIW/e+YJ3rwp7XGej78t4/qY+5Kjd5l9exfPZRaPimMtV"
    "8w9pzqaEmY/jWOJomadjz0O8dObqLvPdL3jJRbCHhLgv3fuCr1gIOxPNhtfx3HcF7FEvzF3fxPNTi2FPrG/ZjbGHRgchXg3k"
    "sm/j+YcgyK6tLBUQu0nHluL8VJRdERfDScghoea38Ak/ZbN3tCxWRrXIbI+YHmwVY5MPL7OdVfeYT08iPa91hH1dqZhyTyM9"
    "N62Avbzzvz6fiiDW6zxiDj6L9HxSFPLycv/ak1cippmeMS9fRnoeWQM5sdJ//B8jXl3axpjeRHre/huyZ/V/7Q3NiN0wr5iR"
    "XyI9s3XIAxv8ax89FrG70CsmOi/S0yUYcvTP/9rrFcP+G/rHPGNRnuuqQH7v+q/d9Zqfhf3sH7OiUJSnc6naFlb9P2s5/Rv2"
    "vPqImFLFojw7jIBcXvnXftcV+z9uTEzXklGePzaE7MP/tesnsLeZE2L2OUd5FjkE+Xrbf+0jN9v2f0bMzSpRnsvCIb/w+te+"
    "eyH2ucScmCfVozzj50I29P7XHhiHvb0zN2aQS5TngljI4f3/tY95iffi6fyY2XWiPGc/h3zB91873lOd9V4QU79ulOfTnrb3"
    "xS8m9nMkX9cd+x8VYTZ8ieQj1mL//feY90I+Whp7fs9qnvw1krffin32f2zumhvJx3zA3gYaLBvyI3npNNv70twyviCSd12K"
    "PXRtZzGyKP6qPfbNtY/FX4jiIRfN7iwxwDJfjOLPWmF/UudbvKUo/qUM9sR/uaWhIYqnrsc+dFhvCS4cxd8vx9oXH7K4FYni"
    "v07GeveetFRyiuLR0Vjj01hLNuRjj7Eut8uWokWjeI1uthx1xTLzXQDf3RVryQmyFHofwKeuxloSp1h6fQjgF0piLanzLG8/"
    "BvDRm7CWUcsskZ8C+Mo3WEuHdZameQG81y2sZepBS0p+AF87D2txO2mZxAL5DDesJfqs5aIYyNmtKmbmdM1iKBTIfdtjLU+T"
    "LdsLB/ITVbCWxdmWHk6BfMMWrMX/L8uDEoH8zzCsZW+BZUKpQF5muu3dkWJ+cQ7ku81YywKnmDNlAnnUX1jLzhIxpcsG8hI9"
    "sJYFJWNmvqvJT3XGWirHW4q/r8mXrsJanG5gLTV5UnGs5WkS1lKTr9iAtSzOwlpq8qMvsZZRT7CWmnzTVawlPB9rqcnTp2Mt"
    "B6WYScyFF2mItbwoEnNRdOGe865gLaViDIVc+K3eWIuhfMz2wi58Qi2spUvlmB5OLrzHLqwlrnrM0xIu/PA6rCWzQcyEUi78"
    "1Sys5f3PWIsLXxeDtSxvGnOujAtf+TfWcrBFTPmyLvyLl62mtIwJfxSvU53Kgkx1almrNJ3qVCJkqlNLHsfrVKdOv4vXqU49"
    "gEx1an+7NJ3qVDLk7+qU9bs6Zf2uTlm/q1PW7+qU9bs6Zf2uTlm/q1PW7+qUlerUnYgXOtWpaptf6FSnLkGmOvV+/wud6lTN"
    "Ay/07+qUlepU3yfx+nd1yvpdnbJSnRq39YVOdepQ1Aud6lTjv+J1qlNT/o7Xv6tTVqpT7Z7F69/VKet3dcr6XZ2yUp1qrqXp"
    "VKe8t7/QqU7N3vFCpzpV/nm8TnXK80W8/l2dsn5Xp6xUpyJ2vtCpTg16Ga9Tnfr5Vbz+XZ2yflenrFSnPHe/0KlOLdn7Qqc6"
    "Jb2O16lOdXkTr39Xp6xUp8q8jde/q1NWqlNNEBeqU2/gQ3Xq45PIuO/qURzVo/3PIuOoBqHuxFHd8XwTGUe1ZtiXyDiqL6fy"
    "IuOopqCOxFEdQe2Io9pRolhUHNUL1Ig4qhGoC3FUF1AL4qgWIP/HUf5Hzo+jnD+zTlQc5Xnk9jjK7Wc/R+qU26UvkTrl9j2Q"
    "Kbf/9jVSp9zeJTdSp9y+Pj9Sp9w+riBSp9zuzqJ0yu1+QpROuX2eGKVTbveSonTK7Q0MUTrl9omFo3TK7a2KROmU2ys6RemU"
    "27MgU253KhqlU26f8S5Ap9wuvQ/QKbf3/BCgU27P+RigU27f+ilAp9zeJC9Ap9yenB+gU26fyAJ1yu3nxUCdcrtUKFCn3B5V"
    "OFCn3N7VKVCn3H6vRKBOuX1cqUCdcnsj50CdcvvpMoE65faSZQN1yu0z3tXUKbcXfV9Tp9ze80NNnXJ7zseaOuX2rZ9q6pTb"
    "m+TV1Cm3J+fX1Cm3T0RCpdx+XnTRKbdLhVx0yu1RhV10yu1dnVx0yu1PSrjolNvHlXLRKbc3cnbRKbefLeOiU24vW9ZFp9w+"
    "pFsKjxuykLOp/xhnud3lcWXx/E615aZuaXxy2dWchZWWZ7nd5j/b7FH3jaVWp3CXZ/A/+sX4418pvMT4UM5C8o1rO6by5s8h"
    "+wjymJOpfMywRVi7JD9vkMaXpkJOLSQ/3J7Gx7RfjHfNSQ6qnc5Tj0KOLiY/3J3ON5RfwplHSXlfkwzeYyLk66XlNjEZfN1V"
    "yKPKylqXTJ5UZSmerYJc524mbzME8sFK8uhJWfzPbZB9qsp+xbP5vTuQ31eXn23P5uZ8yMtrykd++8qrl8fzF14mDzskmMSh"
    "eH7vcDnVLJmO2ux7d8jH8E00ojzWO2KT3KR4Lq/BYG+8XA7fnsvbpeI+3itkX/c83nIH5HEr5XE38vj2oZBnr5LLDM7nDapB"
    "Dg2TL+bk88xreOYFq+Vyswr44mDIk9fIF0Rmiq0A2XetfKsNM734A2s3rpNHL8C8HSAXWy+3wHfWhDTs1bX1svULMzUfDnnm"
    "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"
    "9iS3pGVk+zQ+bDLkqaUtXe6k8gUC5ERny8dxkBdhPw1lLfVLpvLFTpCdylmsZ1J47ALEIrWcZabbXZ04z8HdUnTiQme63daJ"
    "83R1S9OJC32+PVsnXtG/eLZOfOOYSVk68ZB172bqxE96dsnUibdsG5OhE5+5v0mGTjzno93pOvGfE2un68SLPtqephNf+qJB"
    "mk486tiTqTrxq+s6purEuzb4K0UnPrb06hSdeNphhwQrcZ5//PZVJy70mJlZifNMNUtW4kI7LhesxCtOGSxYiW+c6ipYiYe0"
    "fmFW4idb6MxKvOXoBbinnc+81YZZiee8IDIr8Z8VZhXoxItezsnXv/Glg/N14lEn3MjTiV/1d8/TiXfdtD1XJz62afFcnXja"
    "sI2ilTjPz6tFa4idC124oLCV+M/yB0XrEjuvuMYsWrva+cYql0TrSzsPWe6maN1h5yfPJYvWZ3besl26aG1r5zPjMkXrBTvP"
    "6ZElWs/Z+c/YDNF6xM6Ltk4VraXsfOnZP0VrczuPWiFetDax86t1dTynnXc9ckK0Rtn52C67ROtFO0+7Y3ayTpxnQGCyTlzo"
    "4nUJOvGfG/sk68QrLnVL1olvHFkzWSce0rd4sk785PuvSTrxlhk5STrxmY/+TtKJ53z3JEkn/lN4mqQTL1rpZZJOfGnPj0k6"
    "8ahPpGSd+NXaFZN14l0X/pKsEx/7d6dknXjaPlO+6sR5Ru0WrMSFHrBIVuI8Ey3MSlzo9hK5OvGKUTtydeIb03A2iIdUcGaI"
    "n3Txz9eJtyx4la8TnzlseoFOPOcrxqzEf042MSvxovPmMivxpWfO4pzbedSDH5mV+NW4RoKVeNeKAwUr8bHGUMFKPG3TA4KV"
    "OM/IaV914kL3xzArcZ43YyQrcaHTlgpW4hWL+AtW4huHNRGs33hIvHfET96MY1biLXvhvSM+s2hbZiWes63ErMR/bp9doBMv"
    "evttvk586bqh+TrxqP1v5enEr+7zyNOJd3XdnasTHyuWztWJpzX6pujEedYy3tWJCx1rTNOJ86xlvK0TF7ouMkUnXnHE+xT9"
    "G9/YK1UnHlKLS9WJn9zaLE0n3nLugTSd+Eynn9J14jnnHkrXif8c1CJDJ160hp6hE1/q1SNTJx415EGmTvzqj1OydOJdw0tm"
    "68THxu3M1omnHeNxVyfO83T/FL2MnQsd43FbJ87TxSNNJy70n93ZupOdV9xdOlv3sfONBVOz9FN2HrLSo0yd2fnJrV6Z+kE7"
    "b1nFmqGXt/OZBVhLTzvPeQFrPGXnPx83TNc/23nRd/vT9HF2vjSyaZpezs6jXo1N1Y/b+dXdPVP1F3be9f2bFN3Lzsdu2Zyi"
    "b7bztO5rRCtxntXDRStxobNDC1uJ/9yHvES8YrWTopX4xt+Rx4iHDEV+I36y/m3RSrzlFuRD4jMNyJ/Ec45E/iT+8zrsxIv+"
    "gnxLfOkM5GHiUSNviFbiVwMu4j523vXQGdFKfOzz/aKVeNqm05N1Lzvn+fOiZJ240NnhCTrxn818kvWHdl5xVEvkTzvfeKRG"
    "sr7TzkM2KpGsO9v5yYH5Sbq/nbfMeZ+kL7PzmSdeJ+mBdp5z+askfaCd/xz/Jknvb+dFB39K0qfY+dK5yJkF6/7DoxaUTdaT"
    "7fxqwx+T9Td23nWGJ/KnnY+95pes77PztNqMrzpxno/+EKzEhUaelazEeV5A7iIuNN45VydescveXJ34xl+1PJ14yJFJeTrx"
    "k61G5uvEW1b/J18nPjNqfoFOPGe9wsxK/KdXB2YlXrToYmYlvvTjBeQiO4+6O59ZiV+d21ywfuNdhwtW4mOjwgQr8bT9uqfo"
    "/e2cZxH1rk5cqLeaphPnWUS9rRMXGrwmRV9j5xU/PE3RR9v5xsudU/Whdh6y4HSq/srOT2Y1StNd7bxlk91p+jA7n1mmfrpe"
    "2c5zJu1P17vY+c/Hv2bo2+y86KW4DH2ZnS8V8f5G2XnUWnivy9j51Y3TsvSVdt71jHO2nm/nY4fvzdbf23nayYjdEjvnuWjW"
    "V5240K2xzEqc5/lYyUpcaF/szwQ7rzgX+1bYzjfOwX7esfOQl7DPXe38ZMuLzLrazluORFzu2vnM64jXBTvPGYM4muz8p7iw"
    "QCde9MinfH2EnS/9OCpfr2bnUbvdydO72flVD56nE+8avD9XJz7WqVyuTjxt+a2itZCd8zy7XrQ+t3OhgYsLW4n/XHJUtKbZ"
    "ecXPsaLV1c43Bl0TrbftPORA5JNOdn7yS5po3W7nLYOz4WPnM+/dQz9m5zm73xetxH9euCtaiRdVkGeILz14R7QSj/oZuSXc"
    "zq8WXEBPaOddQ6JF6147H1txn2gdaOdpe3je1Ynz/KtXir7fzoX28LytE+dZxDNNJy40Zn+2vsfOKzYul62Ps/ONLWdm6Tft"
    "POTLJ5nodf/DTxbyydRP2XnLQhcy9Cg7nzm8VYZ+1s5zRhxO16fa+c86qIn97bxoGOrCBTtfusY1TT9m51GLnU3Vw+386i9e"
    "qXorO+9qfoX3xc7HbgpP0WPsPG1CaLJOnGfZKck6caHDNifoxH+eQl4iXjEc+Yr4xqHIY8RDtiuXrBM/eQl5j3jLDciHxGeG"
    "Ik8SzzkD+ZP4zwDkVeJFxyDfEl+6JC9JJx6Vob8lfrXGD8iTdt51aPNknfjYfV7JOvG0v8z5qhPn+XafYC1v50I3nJOsxHla"
    "zjErcaGTyufqx+284ugD6D/tfOPvOP/d7Dzkq+Q8PcPOT1qRP4/aecuwj/l6Qzuf+XBugT7cznPuMDDrMTv/WasdsybYedHa"
    "CzGvnS/tim+rYXYe9Vf0fgfs/KqMnnCmnXdd4CdYx9j52E1LBCvxtBVjBStxnr/N/aoTF7oefSNxnuY4yUpc6JcIwUq8YuNA"
    "wUp8o6cqWImH3FFEsBI/6Z7ArMRbRqxiVuIzjd2ZlXjOgBJYo53/dFpVoBMvOkEo0IkvXRKYrxOPeiszTyd+9c92eTrxrvxI"
    "rk58LK+UqxNP235kik6c5wd+VycuVDOl6cR5fuDo0+xc6MK9KTrxisa8FJ34xh6+qTrxkNlXUnXiJ+vJaTrxljnH03TiM/2b"
    "pevEc144ma4T/7lMydCJF3W/ihph50sH9M3UiUed8Qw1ws6vlp+TpRPvOr1Ctk587MaD2TrxtGqbuzpxnj5tUnTiQtU2t3Xi"
    "PN+3TtOJC/U7nK0TryhXzNaJb5w+O0snHpI/zdSJn2yPPEO8ZeXzGTrxmY3RfxLPKRxI14n/LFMvXSde9PWONJ340q4N0nTi"
    "UUsfT9WJXx3ZNlUn3rXS/RSd+FinJSk68bRd9ohW4jyLrBCtxIUOXVrYSvxnDvIw8YoT8M1OfOP9RNFKPKQFfSbxkwrqAvGW"
    "a1ALiM9MeiBaiecsD5n4zz6oEcSLbkL/SXzpbdQd4lH/Rj0iftVyTrQS71rimGglPnbaNtFKPG3zqGSdOM9u45J14kI7RSXo"
    "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"
    "PVzb2J/jdxezbGBM6/TwjFx0Xxnt8c3bbs/2ltGWVX/oZrN/LShQJ34N1kpVqaIVzwvWdleuor39ac6/9v4N/3N9WnOEthK/"
    "z6s14psfjbNdy17ro1WtWkXbE9/nmx+Ns10nHXXVllWropU85vrNj8bZrvXDamulqlfRYlbX/uZH42zXjKTy2kr8Hnyn/Dc/"
    "Gme77n6H9f1QReMf2Dc/Gme7jh+aoebh98gRGSr50Tjbddgf59QZNapo4SfOffOjcbbr+5ztatGaVbSf3m//5kfjbNdP1Ruq"
    "k1yqaD3qNPzmR+Ns1zY7nirv8fujI08V8qNxtqtnw6HKjFpVNKfWQ7/50TjbldXykv/B7yFmb5n8aNzE5VtUEXGctWyLeuBI"
    "GW1hWoQajuuM9Ih/7W9+OiMXO9hOo3NM57aJeZfb8wPttK4ThmmpwiStAQvR1kuTtHsFc7RpTedoD8YP0842maNZ8Xu3+iO0"
    "Efi9RIMR3/xoXL+AYVpgYh/tAOyuf/b55kfjzgXC75SrNlwI0TZGu37zo3Ftg4ZpSzbU1nbi9w7htb/50bh4/K5llNcewF6Q"
    "Wf6bH41bNHGYVuwL00xiiJbwlX3zo3HuwcM018AMNQK/RwdnqORH487h9+ax59Qs2E/o57750bi2k4Zpp/K2q7WkEE0Vdnzz"
    "o3EnfhumrWzZUE3H76FKw29+NK7p5GFahztPlUqGEM3z0VOF/Gjcafwevmqo0h+/5xwY+s2PxjWfMkxjoX3kjbCzjv1k8qNx"
    "mxZvUfcea/ctD1He2XknQl0Pe99RZ2Rf1uLbOaC46xVPunnC/syrs6YxRzwp7jNgd2k8R/P17qwdr+uIJ/nRuHf4Pe2GI57k"
    "R+MG+nTW1JOOeJIfjcvG7+/XOeJJfjTOr1dnbU+aI57kR+NcenfWgj454kl+NC4BvwvjHfEkPxrXo09n7YvZEU/yo3Fp+H3p"
    "l+3f4kl+NK5rv85aeVdHPMmPxl3E74uvOuJJfjTOpz/2ebIjnuRH427hd/a6l0zngPxo3OolW9S8wi00OgcU900pEeoz2M/t"
    "OiOf8Knz7RxQ3L2DL7ith316i2bazvzgb/GkuDvDvvxn5H9cq9ZxxJP8aNzsls00U4IjnuRH4160aqZtP+6IJ/nRuGC3Zlrf"
    "tY54kh+Ne4ffS6U64kl+NO6gezPtzkdHPMmPxo00NtMSRzviSX407pHt99OOeJIfjRsjN9N8/nHEk/xoXJpHM+1KQ0c8yY/G"
    "9VLwe4wjnt/87OMe4/epAx3xJD8aN0RtpoX08JHpHJAfjQtdukXtNaCO9u0c2OMelhqhmmCnuk/ngOJOfQDVc4onxZ36gG/1"
    "3B5P8qNxVM8pnuRH46ieUzzJj8ZRPad4kh+No3pO8SQ/Gkf1nOJJfjSO6jnFk/xoHNVziif50Tiq5xRP8qNxVM8pnuRH46ie"
    "UzzJj8ZRPad4kh+No7pP54D8aBzVfToHFHfqA6hvpD6U+k7qI6lvpD6U+k7qI0X0DfLJSuqpT+vcTuL6x+tVboN+Yup+9KUP"
    "GzJ1CerO4U2V1DWoO19w9cR1R2mmtrDVI1w9Ma/JY5iSvTPEraY6TLl1YJSbPLG+uwH98OnFdd1VzPvx1lClHeZrcnuoMhv9"
    "7/u08W6/oR9eETfK7YB9Hpr39ZLFbv2uW5TyWNeYeIvSkuaxz3sF93m6waLE4r5euFL/S/3udf22m/vhTLUe3ouvhzLV7/tY"
    "W7+780aCajv3KQkJKvW51Mfa/Mh++3qCGmLvc6mPtfmRPTY+Qd1h73Opj7X5kf3ZlQSV+lzqY21+ZM++nKBSn0t9rM2P7BmX"
    "ElTqc6mPtfmRffzFBJX6XOpjbX5kzz+foFKfS32sze+b3ZqgUp9Lfey/fnb7az1BpT6X+libH9mLnktQqc+lPtbmR/bKsQkq"
    "9bm2Ppb8yP4lJkH9ZBr6rf8lP7IzXG3vBfW/1O/azn16VKZqey+O4+rS6Iyc1324OuLWLrepPYZ/O8d0rkeuyFRs/UxkWKZi"
    "63s+NDjRYhb637/rHzhl63O9l2QqndDPkB/Zd+T0Om3rc6sszFQmoZ8hP7IfXHT4tK3PPT0tU0lCP0N+ZI/5+dfodehzf/st"
    "U7H1M+RH9sJtekbb+tyJQZmKrZ8hP7IP2REcnYc+d8/YTKUo+hnyI3uxnqujb6DP/TA8U2mGfob8yF5+9p5oM/rce4MzFVs/"
    "Q35kv/HwdPRh9LlHB2Uqtn6G/Mh+8PmVaFuf6+GdqYShnyE/spfd/zF6MfrcQd0ylWHoZ8iP7Ppj8UwH9LmsU6Zi62fIj+zX"
    "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"
    "2EmPcmmtztnosJMu322tjtnqsJMeENlGXbzNYSd9nH9b9fF2h530gFrt1Pnf2Uk/eb+dOijKYSd9fFR7dcxmh530LQM7qGs2"
    "OOykF6/QUT260mEn/ePljurKBQ476bMndlKvT3HYSY8p2VmdPdJhH4647HPx08yIUy3VT5vZ1139ZZiflos4Hljlp40a4qZ6"
    "nfPTHiHOA1DfwkNaqRrq26lKrdQI1LdZJ1qqLhH+mqlPS3VFir8WxFqqhyoM1rYebaF26jVYWzGqheq6YbDm0xDnKXWwtvJ9"
    "czWp6hDt7+vN1SuDhmgF0c3V9MghWmO9sdr4s5+2+kxzVfcepHUq1Vl16Yq8GNxJ3eThq7Uf3FZ9kNT/m/76Skf1YBOHnXSt"
    "Ykd1WL3v/O16Sd8O6rAfHHbSK29vrwZVcthJ3/qwnTqmnMNO+vo67dSWZRx20m8Maatec3bYSS+9vY1au6zDTnrFB61VXsFh"
    "J/10ndZqs6oOO+nPR5rU4rUcdtJfHeHqq4YOO+kheZ6qcwuHnfR5PT3VmyaHnfT70Zpazsdh3/u5hxq/aqL2NcRLdUPfliV5"
    "q07o477M9Vb90NfV/+qtLkCfZwzoqWr4TnTO7qlWXBGkTW7vo47rFaQ1OuCjrqoWpGU69VLP3Q3Uqvr3Upshv/Y42kutjvwS"
    "86mXehT5prVbb7UB8k/X8b3VROSj2ut7qy4DA7Qf9/VWG/+3Js47LKrr6eMLC6jYEYKKBXtBjZG6vnfmCki1IWVB2lJUlEQQ"
    "BP2hQZSFUJWyimLFrqixYEEwGjvYBSVq1BC7EkBEBETwPXmenT356z6f7z33njlnd8/cMzN3+0eKTQX++OfvUaJqvxydw2LE"
    "p68ScaJJpHixPhEDAiLF5VIlKgsixd8GKvF/LyJFS1Bi6YjFomGwEtvnLxbvpirx297FYucTSsx8tVhc8lyJUUOjxMd9kjAm"
    "IErs45yE8rwo8dvKJLx2kz0nFydhqCRazP6UhLsmRos9zZPxgn+0+M+SZFyijBY7TiVj3o5oUcXscFkQI86pTMAt7Ll3VaEc"
    "1xWEiDl5ctwYyJ7fIuRYZRIi1sjkOO9RsJj8xQt/38CeW4+zefAOFseHeGGzcbBo3NUL7R8HiTmHPdFpW5Co5+qJwtwg8Va1"
    "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"
)

MESH_WIDTH = 72.42881774902344    # model units across the film width (35 mm)
MESH_PITCH = 39.58382606506348    # model units per frame (4-perf pitch)
MESH_IMG_ASPECT = 50.988 / 38.439  # aspect of the picture area (≈ 1.3265)
HOLDER_SLOT = 0
IMAGE_SLOT = 1

COLLECTION_NAME = "Film Reel"
ROOT_NAME = "Film Reel"
MARK = "film_reel_addon"
SERIAL = "film_reel_serial"
RECOMMENDED_MAX = 240

FPS_ITEMS = [
    ('8', "8", ""), ('12', "12", ""), ('16', "16", ""),
    ('18', "18", ""), ('23.976', "23.976", ""), ('24', "24", ""),
    ('25', "25", ""), ('29.97', "29.97", ""), ('30', "30", ""),
    ('48', "48", ""), ('50', "50", ""), ('59.94', "59.94", ""),
    ('60', "60", ""), ('CUSTOM', "Custom", ""),
]

# (id, label, film width in mm). Every size uses the Film35mm model scaled
# uniformly so its width equals the film width.
FILM_SIZES = [
    ('8MM', "8mm", 8.0), ('SUPER8', "Super 8", 8.0), ('16MM', "16mm", 16.0),
    ('SUPER16', "Super 16", 16.0), ('24MM', "24mm", 24.0), ('35MM', "35mm", 35.0),
    ('46MM', "46mm", 46.0), ('60MM', "60mm", 60.0), ('65MM', "65mm", 65.0),
    ('70MM', "70mm", 70.0), ('IMAX70', "IMAX 70", 70.0),
]
FILM_SIZE_ITEMS = [(i, l, "") for i, l, _ in FILM_SIZES]
FILM_WIDTH = {i: w for i, _, w in FILM_SIZES}
FILM_LABEL = {i: l for i, l, _ in FILM_SIZES}

# Picture shapes (width / height) the image can be shown in on each film frame.
# The picture is centred in the frame's picture window; whatever the window has
# left over is black (letterbox / pillarbox), exactly like a real exposure.
ASPECT_PRESETS = [
    ('1_1', "1:1 Square", 1.0),
    ('4_3', "4:3 (1.33)", 4.0 / 3.0),
    ('ACADEMY', "Academy (1.37)", 1.37),
    ('3_2', "3:2 (1.50)", 1.5),
    ('1_66', "Euro (1.66)", 5.0 / 3.0),
    ('16_9', "16:9 (1.78)", 16.0 / 9.0),
    ('1_85', "Flat (1.85)", 1.85),
    ('2_00', "2:1 (2.00)", 2.0),
    ('2_35', "Scope (2.35)", 2.35),
    ('2_39', "Scope (2.39)", 2.39),
    ('2_76', "Ultra (2.76)", 2.76),
    ('3_4', "3:4 Portrait", 0.75),
    ('9_16', "9:16 Vertical", 9.0 / 16.0),
]
ASPECT_VALUE = {i: v for i, _, v in ASPECT_PRESETS}
ASPECT_ITEMS = (
    [('RECOMMENDED', "Recommended",
      "Match the loaded footage, so the picture is never stretched or cropped"),
     ('WINDOW', "Full Frame (1.33)",
      "Fill the whole picture window of the film frame (the original behaviour)")]
    + [(i, l, "") for i, l, _ in ASPECT_PRESETS]
    + [('CUSTOM', "Custom", "Enter any width / height ratio")])

# Longest-side texture sizes offered for image sequences (0 = original size).
RESOLUTION_STEPS = (4096, 2048, 1920, 1280, 1024, 512, 256)
RESOLUTION_LABELS = {4096: "4096 px (4K)", 2048: "2048 px (2K)", 1920: "1920 px (Full HD)",
                     1280: "1280 px (HD)", 1024: "1024 px", 512: "512 px", 256: "256 px"}
RESOLUTION_ITEMS = (
    [('RECOMMENDED', "Recommended",
      "The source's own size, stepped down only as far as needed to keep the frame "
      "textures within about 1 GB of memory (never upscaled)"),
     ('0', "Original", "Keep the original resolution")]
    + [(str(s), RESOLUTION_LABELS[s], "") for s in RESOLUTION_STEPS])
MEMORY_BUDGET_GB = 1.0   # what "Recommended" resolution aims to stay under


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------
def decode_mesh():
    raw = zlib.decompress(base64.b64decode("".join(_MESH_B64.split())))
    nv, npoly, nl = struct.unpack_from('<III', raw, 0)
    o = 12
    verts = struct.unpack_from('<%df' % (nv * 3), raw, o); o += nv * 12
    sizes = struct.unpack_from('<%dH' % npoly, raw, o); o += npoly * 2
    idx = struct.unpack_from('<%dH' % nl, raw, o); o += nl * 2
    mats = struct.unpack_from('<%dB' % npoly, raw, o); o += npoly
    uvs = struct.unpack_from('<%df' % (nl * 2), raw, o)
    vlist = [verts[i:i + 3] for i in range(0, len(verts), 3)]
    faces, p = [], 0
    for s in sizes:
        faces.append(tuple(idx[p:p + s]))
        p += s
    return vlist, faces, list(mats), list(uvs)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def expand_sequence(directory, names):
    """One selected file like frame_0001.png -> all siblings frame_NNNN.png."""
    if len(names) == 1:
        m = re.match(r'^(.*?)(\d+)(\.[^.]+)$', names[0])
        if m:
            pre, _num, ext = m.groups()
            pat = re.compile('^' + re.escape(pre) + r'\d+' + re.escape(ext) + '$', re.I)
            try:
                found = [f for f in os.listdir(directory) if pat.match(f)]
            except OSError:
                found = []
            if len(found) > 1:
                names = found
    return sorted(names, key=natural_key)


def movie_extensions():
    try:
        return set(bpy.path.extensions_movie)
    except Exception:
        return {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mpg',
                '.mpeg', '.wmv', '.mxf', '.flv', '.ogv'}


def spiral_table(r0, b, s_max):
    """Arc length -> angle table for the Archimedean spiral r = r0 + b*theta."""
    dth = 0.002
    n = 20000
    while True:
        th = np.arange(n + 1, dtype=np.float64) * dth
        r_mid = r0 + b * (th[:-1] + dth * 0.5)
        ds = np.sqrt(r_mid * r_mid + b * b) * dth
        arc = np.concatenate(([0.0], np.cumsum(ds)))
        if arc[-1] >= s_max:
            return arc, th
        n *= 2


def _strip_coords(verts, scale, n):
    """Shared set-up for every strip layout: per-vertex arc length along the strip,
    film-width coordinate x and film-thickness coordinate z, for n frames."""
    V = np.asarray(verts, dtype=np.float64) * scale
    nv = len(V)
    y_min, y_max = V[:, 1].min(), V[:, 1].max()
    pitch = float(y_max - y_min)
    y_local = V[:, 1] - 0.5 * (y_max + y_min)
    z_local = V[:, 2].copy()
    # The model's end edges bow by a hair (~0.007 model units) and don't quite
    # mirror each other. Snap the end-edge vertices onto exact, flat edge lines
    # so one frame's tip edge coincides exactly with the next frame's bottom edge.
    half = 0.5 * pitch
    end_top = y_local > 0.985 * half
    end_bot = y_local < -0.985 * half
    y_local[end_top], y_local[end_bot] = half, -half
    z_local[end_top | end_bot] = 0.0
    c = (np.arange(n, dtype=np.float64) + 0.5) * pitch          # frame centres (arc length)
    s_arc = (c[:, None] - y_local[None, :]).ravel()               # local +Y runs against the strip
    x = np.tile(V[:, 0], n)
    z = np.tile(z_local, n)
    return nv, pitch, s_arc, x, z


def strip_geometry(verts, faces, mats, uvs, scale, n, mode, r0=0.0, wind=0.0):
    """Combine n copies of the frame mesh into ONE continuous strip.

    Frame k covers arc length k*pitch .. (k+1)*pitch, where pitch is the exact
    Y extent of the frame mesh, so the tip edge of one frame is the bottom edge
    of the next (identical vertex positions). In SPIRAL mode every vertex is
    bent onto the spiral, so the strip stays seamless instead of leaving
    wedge-shaped gaps between flat frames.

    Returns (positions Nx3, faces, loop normals Lx3, material index per poly
    (0 = holder, k+1 = image of frame k), uv floats).
    """
    nv, pitch, s_arc, x, z = _strip_coords(verts, scale, n)

    if mode == 'SPIRAL':
        b = wind / (2.0 * math.pi)
        arc, th_tab = spiral_table(r0, b, float(s_arc.max()) + pitch)
        th = np.interp(s_arc, arc, th_tab)
        r = r0 + b * th
        ty = b * np.cos(th) - r * np.sin(th)
        tz = b * np.sin(th) + r * np.cos(th)
        ln = np.hypot(ty, tz)
        ty, tz = ty / ln, tz / ln
        ny, nz = tz, -ty                                          # outward normal
        py = r * np.cos(th) + z * ny
        pz = r * np.sin(th) + z * nz
        vnorm = np.stack([np.zeros_like(ny), ny, nz], axis=1)
    else:
        py = -s_arc
        pz = z
        vnorm = np.tile(np.array([0.0, 0.0, 1.0]), (n * nv, 1))

    pos = np.stack([x, py, pz], axis=1)

    base_faces = [tuple(f) for f in faces]
    all_faces = []
    for k in range(n):
        off = k * nv
        all_faces.extend(tuple(i + off for i in f) for f in base_faces)
    loop_v = np.fromiter((i for f in all_faces for i in f), dtype=np.int64)
    lnorm = vnorm[loop_v]

    mats = np.asarray(mats, dtype=np.int64)
    mat_idx = np.concatenate([np.where(mats == IMAGE_SLOT, k + 1, 0) for k in range(n)])
    uv_all = np.tile(np.asarray(uvs, dtype=np.float32), n)
    return pos, all_faces, lnorm, mat_idx, uv_all


_SPIRAL_CACHE = {}


def _spiral_cached(r0, b, s_max):
    key = (round(float(r0), 12), round(float(b), 12))
    ent = _SPIRAL_CACHE.get(key)
    if ent is None or ent[0][-1] < s_max:
        if len(_SPIRAL_CACHE) > 16:
            _SPIRAL_CACHE.clear()
        ent = spiral_table(r0, b, s_max)
        _SPIRAL_CACHE[key] = ent
    return ent


def _frame_pitch(verts, scale):
    """Arc length of one frame along the strip (the exact Y extent of the frame mesh)."""
    ys = np.asarray(verts, dtype=np.float64)[:, 1]
    return float((ys.max() - ys.min()) * scale)


def unwind_geometry(verts, scale, n, r0, wind, r0t, path, c_t, progress, probe_s=None):
    """Positions of every strip vertex while the film unwinds along a path.

    Everything happens in the reel's own plane (local Y/Z; the reel axis is local X):
      * supply reel   centred on the origin, core radius r0, the original wound spiral
        (frame 1 at the core). Film pays out from its OUTER end (the last frame).
      * take-up reel  centred on c_t, core radius r0t. The leading end (last frame) is
        fixed to its core and the film winds on over it.
      * path          polyline (m x 2) the free film follows, first point = start.
        Between the reels' current outer radius and the path ends the film runs
        along straight tangent lines, so it always leaves / meets a reel tangentially.
    progress 0 = everything on the supply reel, 1 = everything on the take-up reel
    (linear in film length fed through).

    Returns (positions Nx3, per-vertex normals Nx3, warning string,
    total film length fed from progress 0 to 1).
    """
    if probe_s is not None:
        # probe mode: only evaluate the given arc-length positions (film centre line, z = 0)
        pitch = _frame_pitch(verts, scale)
        s = np.asarray(probe_s, dtype=np.float64).ravel()
        x = np.zeros_like(s)
        z = np.zeros_like(s)
    else:
        nv, pitch, s, x, z = _strip_coords(verts, scale, n)
    L = n * pitch
    b = wind / (2.0 * math.pi)
    eps_t = 1.0                                                 # take-up winding direction
    arc_s, th_s = _spiral_cached(r0, b, L + pitch)
    arc_t, th_t = _spiral_cached(r0t, b, L + pitch)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    c_t = np.asarray(c_t, dtype=np.float64)
    A, B = path[0], path[-1]

    def rad_s(o):
        return r0 + b * float(np.interp(o, arc_s, th_s)) if o > 0.0 else r0

    def rad_t(sg):
        return r0t + b * float(np.interp(sg, arc_t, th_t)) if sg > 0.0 else r0t

    def free_curve(rs, rt):
        D = A
        d = max(float(np.hypot(D[0], D[1])), rs)
        phi_s = math.atan2(D[1], D[0]) - math.acos(min(1.0, rs / d))
        Qs = rs * np.array([math.cos(phi_s), math.sin(phi_s)])
        D2 = B - c_t
        d2 = max(float(np.hypot(D2[0], D2[1])), rt)
        phi_t = math.atan2(D2[1], D2[0]) + eps_t * math.acos(min(1.0, rt / d2))
        Qt = c_t + rt * np.array([math.cos(phi_t), math.sin(phi_t)])
        pts = np.vstack([Qs, path, Qt])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        pts = pts[np.concatenate(([True], seg > 1e-9))]
        if len(pts) < 2:
            pts = np.vstack([pts[0], pts[0] + np.array([1e-6, 0.0])])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        dirs = np.diff(pts, axis=0) / seg[:, None]
        return pts, cum, dirs, phi_s, phi_t

    warn = ""
    if float(np.hypot(A[0], A[1])) < 0.98 * rad_s(L):
        warn = "Path start is inside the full reel - move it outward"
    elif float(np.hypot(*(B - c_t))) < 0.98 * rad_t(L):
        warn = "Path end is inside the take-up reel at full size"

    G_end = free_curve(r0, rad_t(L))[1][-1]
    q = min(1.0, max(0.0, float(progress))) * (L + G_end)       # film fed through so far
    o = L - q                                                    # film still on the supply
    rs = rad_s(o)
    sigma = 0.0                                                  # film already on the take-up
    for _ in range(10):
        pts, cum, dirs, phi_s, phi_t = free_curve(rs, rad_t(sigma))
        new = min(L, max(0.0, L - o - float(cum[-1])))
        done = abs(new - sigma) < 1e-9 * max(1.0, L)
        sigma = new
        if done:
            break
    rt = rad_t(sigma)
    pts, cum, dirs, phi_s, phi_t = free_curve(rs, rt)
    G = float(cum[-1])

    N = len(s)
    py = np.empty(N)
    pz = np.empty(N)
    ny = np.empty(N)
    nz = np.empty(N)
    m_sup = s < o
    m_tak = (s > L - sigma) if sigma > 0.0 else np.zeros(N, dtype=bool)
    m_free = ~(m_sup | m_tak)

    if m_sup.any():                                              # supply spiral (rotated so its
        th = np.interp(s[m_sup], arc_s, th_s)                    # outer end sits at the exit point)
        r = r0 + b * th
        phi = th - float(np.interp(o, arc_s, th_s)) + phi_s
        c, sn = np.cos(phi), np.sin(phi)
        ty, tz = b * c - r * sn, b * sn + r * c
        ln = np.hypot(ty, tz)
        ty, tz = ty / ln, tz / ln
        ny[m_sup], nz[m_sup] = tz, -ty
        py[m_sup] = r * c + z[m_sup] * tz
        pz[m_sup] = r * sn - z[m_sup] * ty

    if m_tak.any():                                              # take-up spiral
        th = np.interp(L - s[m_tak], arc_t, th_t)
        r = r0t + b * th
        phi = phi_t + eps_t * (float(np.interp(sigma, arc_t, th_t)) - th)
        c, sn = np.cos(phi), np.sin(phi)
        den = np.sqrt(r * r + b * b)
        ty = (-b * c - eps_t * r * sn) / den
        tz = (-b * sn + eps_t * r * c) / den
        ny[m_tak], nz[m_tak] = tz, -ty
        py[m_tak] = c_t[0] + r * c + z[m_tak] * tz
        pz[m_tak] = c_t[1] + r * sn - z[m_tak] * ty

    if m_free.any():                                             # free run: bridge, path, bridge
        u = np.clip(s[m_free] - o, 0.0, G)
        idx = np.clip(np.searchsorted(cum, u, side='right') - 1, 0, len(dirs) - 1)
        ty, tz = dirs[idx, 0], dirs[idx, 1]
        ny[m_free], nz[m_free] = tz, -ty
        py[m_free] = np.interp(u, cum, pts[:, 0]) + z[m_free] * tz
        pz[m_free] = np.interp(u, cum, pts[:, 1]) - z[m_free] * ty

    pos = np.stack([x, py, pz], axis=1)
    vnorm = np.stack([np.zeros(N), ny, nz], axis=1)
    return pos, vnorm, warn, L + G_end


def align_progress(verts, scale, n, r0, wind, r0t, path, c_t, k, target, coarse=480):
    """Unwind progress at which the centre of geometric frame k (0-based) comes closest to
    `target` (a point in the reel plane: local Y, Z). Returns (progress, distance).

    The frame centre's route over progress 0 -> 1 is: round the supply reel, along the free
    run (the path), round the take-up reel. A coarse scan finds the best passage, then the
    neighbourhood is refined. If the film passes the target more than once (the frame circles
    the supply reel once per turn), the earliest passage is used."""
    pitch = _frame_pitch(verts, scale)
    probe = np.array([(k + 0.5) * pitch])
    target = np.asarray(target, dtype=np.float64)

    def dist(p):
        pos = unwind_geometry(verts, scale, n, r0, wind, r0t, path, c_t, p, probe_s=probe)[0]
        return float(math.hypot(pos[0, 1] - target[0], pos[0, 2] - target[1]))

    ps = np.linspace(0.0, 1.0, coarse + 1)
    ds = np.array([dist(p) for p in ps])
    i = int(ds.argmin())
    best_p, best_d = float(ps[i]), float(ds[i])
    lo, hi = float(ps[max(0, i - 1)]), float(ps[min(coarse, i + 1)])
    for _ in range(5):
        cand = np.linspace(lo, hi, 21)
        cd = [dist(p) for p in cand]
        j = int(np.argmin(cd))
        if cd[j] < best_d:
            best_p, best_d = float(cand[j]), float(cd[j])
        lo, hi = float(cand[max(0, j - 1)]), float(cand[min(20, j + 1)])
    return best_p, best_d


def pitch_mm_for(film_size):
    return MESH_PITCH * FILM_WIDTH[film_size] / MESH_WIDTH


MM_TO_M = 0.001
# Smallest radial growth per turn allowed on a wound reel. Real film is ~0.14 mm thick; much below
# this the turns lie on top of each other and the reel collapses into one z-fighting hoop.
WIND_MIN_MM = 0.2


def base_units_per_mm(scene):
    """Blender units that equal 1 mm of REAL film in this scene.

    Blender's default is 1 BU = 1 m, so 1 mm = 0.001 BU and a 35 mm film is 0.035 BU wide.
    If the scene's Unit Scale is changed (Scene > Units > Unit Scale) the result follows it,
    e.g. Unit Scale 0.001 (1 BU = 1 mm) gives 1.0, so 35 mm film is 35 BU wide.
    With the unit system set to None, 1 BU is treated as 1 m.
    """
    us = scene.unit_settings
    if us.system == 'NONE' or us.scale_length <= 0.0:
        return MM_TO_M
    return MM_TO_M / us.scale_length


def units_per_mm(scene, fr):
    """Blender units per mm of film, including the optional size multiplier."""
    return base_units_per_mm(scene) * fr.scale_multiplier


def film_width_m(fr):
    """Real-world width in metres the film strip is built at (before Unit Scale)."""
    return FILM_WIDTH[fr.film_size] * MM_TO_M * fr.scale_multiplier


def aspect_text(a):
    """'16:9 (1.78)' for a standard ratio (within 1.5 %), else '1.43:1'."""
    for _id, label, v in ASPECT_PRESETS:
        if abs(v - a) / a < 0.015:
            return label
    return "%.2f:1" % a


def dims_for_limit(w, h, lim):
    """Size after capping the longest side at lim px (0 = no cap, never upscales)."""
    m = max(w, h)
    if lim and m > lim:
        k = lim / float(m)
        return max(1, int(round(w * k))), max(1, int(round(h * k)))
    return w, h


def picture_mapping(src_w, src_h, target_aspect, fit):
    """UV mapping that shows the source image at `target_aspect` on a frame.

    The picture rectangle (aspect `target_aspect`) is the largest one that fits,
    centred, in the frame's picture window (aspect MESH_IMG_ASPECT). Inside it
    the image is stretched over the whole rectangle (fit 'STRETCH') or keeps its
    own proportions and is cropped to fill it (fit 'CROP').

    Returns None when the plain UVs already do the job, otherwise
    (scale_x, scale_y, offset_x, offset_y, clip, mask):
      * the first four feed a Mapping node (texture_coord = uv * scale + offset);
      * clip is True when the leftover window area can simply be left to the
        texture's CLIP extension (picture smaller than the window, image not cropped);
      * mask is None, or (scale_x, scale_y, offset_x, offset_y) mapping uv to
        picture-rectangle coordinates (0..1 inside), for the one case CLIP can't
        handle: a picture that is both cropped and smaller than the window.
    """
    win = MESH_IMG_ASPECT
    ta = max(0.05, float(target_aspect))
    if abs(ta - win) < 1e-4:
        fw = fh = 1.0
    elif ta > win:
        fw, fh = 1.0, win / ta
    else:
        fw, fh = ta / win, 1.0
    sx = sy = 1.0
    if fit == 'CROP' and src_w > 0 and src_h > 0:
        ia = src_w / float(src_h)
        if abs(ia - ta) >= 1e-4:
            if ia > ta:
                sx = ta / ia
            else:
                sy = ia / ta
    letterboxed = fw < 1.0 - 1e-6 or fh < 1.0 - 1e-6
    cropped = sx != 1.0 or sy != 1.0
    if not letterboxed and not cropped:
        return None
    # picture-rectangle coords r = (uv - (1 - f) / 2) / f, texture = r * s + (1 - s) / 2
    mask = None
    if letterboxed and cropped:
        mask = (1.0 / fw, 1.0 / fh, -(1.0 - fw) / (2.0 * fw), -(1.0 - fh) / (2.0 * fh))
    return (sx / fw, sy / fh,
            -(1.0 - fw) / (2.0 * fw) * sx + (1.0 - sx) * 0.5,
            -(1.0 - fh) / (2.0 * fh) * sy + (1.0 - sy) * 0.5,
            letterboxed and not cropped, mask)


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------
class FilmReelFile(PropertyGroup):
    name: StringProperty()


def _fps_value(fr):
    if fr.fps_preset == 'CUSTOM':
        return max(0.001, fr.custom_fps)
    return float(fr.fps_preset)


def yielded_frames(fr):
    if fr.src_frames <= 0:
        return 0
    dur = fr.src_frames / max(0.001, fr.source_fps)
    return max(1, int(round(dur * _fps_value(fr))))


def recommended_frames(fr):
    return min(yielded_frames(fr), RECOMMENDED_MAX)


def effective_range(fr):
    """(first, last) reel frame to build, 1-based and inclusive, clamped to the clip."""
    y = max(1, yielded_frames(fr))
    first = min(max(1, fr.frame_start), y)
    last = min(max(first, fr.frame_end), y)
    return first, last


def effective_frames(fr):
    first, last = effective_range(fr)
    return last - first + 1


def source_aspect(fr):
    if fr.src_width > 0 and fr.src_height > 0:
        return fr.src_width / float(fr.src_height)
    return MESH_IMG_ASPECT


def picture_aspect(fr):
    """Width / height the picture is shown at on each frame."""
    p = fr.aspect_preset
    if p == 'RECOMMENDED':
        return source_aspect(fr)
    if p == 'WINDOW':
        return MESH_IMG_ASPECT
    if p == 'CUSTOM':
        return max(0.05, fr.custom_aspect)
    return ASPECT_VALUE.get(p, MESH_IMG_ASPECT)


def recommended_resolution(fr):
    """Longest-side cap in px for the frame textures (0 = original size).

    The source's own size, stepped down through RESOLUTION_STEPS only as far as
    needed for the chosen frame range to fit in MEMORY_BUDGET_GB. Never upscales.
    Video frames can't be resampled, so they always stay native (0).
    """
    unique = max(1, len(set(source_index_map(fr)))) if fr.src_frames > 0 else 1
    return recommended_limit(fr.source_type, fr.src_width, fr.src_height, unique)


def recommended_limit(source_type, src_w, src_h, unique):
    """Same as recommended_resolution but from plain values (used when swapping footage)."""
    m = max(src_w, src_h)
    if source_type != 'SEQUENCE' or m <= 0:
        return 0
    candidates = [0] + [s for s in RESOLUTION_STEPS if s < m]
    for lim in candidates:
        w, h = dims_for_limit(src_w, src_h, lim)
        if unique * w * h * 4 / (1024 ** 3) <= MEMORY_BUDGET_GB:
            return lim
    return candidates[-1]


def resolved_limit(fr):
    """Longest-side cap actually used for the frame textures (0 = original)."""
    if fr.max_image_size == 'RECOMMENDED':
        return recommended_resolution(fr)
    return int(fr.max_image_size)


def target_dims(fr):
    w, h = fr.src_width, fr.src_height
    if fr.source_type == 'SEQUENCE':
        return dims_for_limit(w, h, resolved_limit(fr))
    return w, h


def render_size_for(fr):
    """Scene render size for the chosen picture aspect, at the chosen resolution."""
    w, h = target_dims(fr)
    long_side = float(max(w, h))
    a = picture_aspect(fr)
    rw, rh = (long_side, long_side / a) if a >= 1.0 else (long_side * a, long_side)

    def even(v):
        return max(4, int(round(v / 2.0)) * 2)
    return even(rw), even(rh)


def source_index_map(fr):
    """Source frame index for each of the frames on the reel.

    The reel covers the chosen Start..End range (1-based, inclusive) of the
    clip, in chronological order, at the chosen frame rate.
    """
    first, last = effective_range(fr)
    ratio = fr.source_fps / _fps_value(fr)
    return [min(fr.src_frames - 1, int(j * ratio + 1e-6)) for j in range(first - 1, last)]


def memory_gb(fr):
    w, h = target_dims(fr)
    unique = len(set(source_index_map(fr)))
    return unique * w * h * 4 / (1024 ** 3)


_LOADING = [False]      # True while a reel's saved settings are copied into the panel


def _start_changed(self, context):
    if _LOADING[0]:
        return
    if self.frame_end < self.frame_start:
        self.frame_end = self.frame_start


def _end_changed(self, context):
    if _LOADING[0]:
        return
    if self.frame_start > self.frame_end:
        self.frame_start = self.frame_end


def _end_a_edited(self, context):
    if _LOADING[0]:
        return
    if self.end_obj_a is not None:      # typing coordinates overrides the picked object
        self.end_obj_a = None


def _end_b_edited(self, context):
    if _LOADING[0]:
        return
    if self.end_obj_b is not None:
        self.end_obj_b = None


class FilmReelSettings(PropertyGroup):
    # --- what the Build Film Reel button will do
    build_mode: EnumProperty(
        name="Build Mode",
        items=[('ADD', "Add New Reel", "Build Film Reel adds an additional reel and keeps the existing ones"),
               ('REPLACE', "Replace Reel", "Build Film Reel throws the existing reel away and rebuilds it from scratch with "
                "every setting as it is in this panel now (position, cores, unwinding, flips and "
                "frame order are all reset)")],
        default='ADD')
    # --- source
    source_type: EnumProperty(items=[('NONE', "None", ""), ('SEQUENCE', "Image Sequence", ""),
                                     ('VIDEO', "Video", "")], default='NONE')
    directory: StringProperty(subtype='DIR_PATH')
    files: CollectionProperty(type=FilmReelFile)
    src_width: IntProperty(default=0)
    src_height: IntProperty(default=0)
    src_frames: IntProperty(default=0)
    source_fps: FloatProperty(name="Source FPS", default=24.0, min=0.001, max=1000.0,
                              precision=3, description="Frame rate of the loaded footage")
    # --- main settings
    fps_preset: EnumProperty(name="Frame Rate", items=FPS_ITEMS, default='24',
                             description="Frame rate the reel is sampled at")
    custom_fps: FloatProperty(name="Custom FPS", default=24.0, min=0.001, max=1000.0, precision=3)
    film_size: EnumProperty(name="Film Size", items=FILM_SIZE_ITEMS, default='35MM')
    frame_start: IntProperty(
        name="Start Frame", default=1, min=1, max=100000, update=_start_changed,
        description="First frame of the reel. Frames are counted from 1 at the chosen frame rate "
                    "(position in the clip, not the number in the file name)")
    frame_end: IntProperty(
        name="End Frame", default=1, min=1, max=100000, update=_end_changed,
        description="Last frame of the reel (inclusive). The reel holds every frame from Start to "
                    "End, in chronological order")
    # --- options
    layout_mode: EnumProperty(
        name="Layout",
        items=[('STRAIGHT', "Straight Strip", "Frames laid end to end in a straight line"),
               ('SPIRAL', "Wound Reel", "Strip wound in a spiral around a core (frame 1 at the core)")],
        default='STRAIGHT')
    stand_upright: BoolProperty(name="Stand Upright", default=True,
                                description="Straight strip: stand it up facing the front view "
                                            "(first frame on top). Off: lay it flat on the floor")
    core_radius_mm: FloatProperty(name="End A (Supply)", default=60.0, min=1.0, soft_max=300.0,
                                  description="Core radius of the supply reel (End A), in mm of film. "
                                              "Used when a new reel is built")
    takeup_core_mm: FloatProperty(name="End B (Take-up)", default=60.0, min=1.0, soft_max=300.0,
                                  description="Core radius of the take-up reel (End B), in mm of film. "
                                              "Used when a new reel is built")
    wind_spacing_mm: FloatProperty(name="Wind Spacing", default=1.0, min=0.2, soft_max=10.0,
                                   description="Wound reel: radial growth per turn, in mm of film "
                                               "(real film is ~0.14 mm; larger looks better)")
    aspect_preset: EnumProperty(
        name="Aspect Ratio", items=ASPECT_ITEMS, default='RECOMMENDED',
        description="Shape of the picture on every frame. The picture is centred in the frame "
                    "window, the rest stays black. Recommended matches the loaded footage")
    custom_aspect: FloatProperty(
        name="Custom Aspect", default=16.0 / 9.0, min=0.2, max=10.0, precision=3,
        description="Picture width divided by height")
    image_fit: EnumProperty(
        name="Image Fit",
        items=[('STRETCH', "Stretch", "Stretch the whole image over the chosen picture shape"),
               ('CROP', "Crop to Fill",
                "Keep the image's own aspect and crop its edges to fill the chosen picture shape")],
        default='STRETCH',
        description="What to do when the footage and the chosen aspect ratio differ "
                    "(no effect while they match)")
    max_image_size: EnumProperty(
        name="Resolution", items=RESOLUTION_ITEMS, default='RECOMMENDED',
        description="Longest side of every frame texture (image sequences only; "
                    "video frames always decode at their native size)")
    pack_images: BoolProperty(
        name="Pack Images", default=False,
        description="Embed downscaled frames in the .blend. Off: frames reload from disk at "
                    "full resolution next time the file is opened")
    # --- custom reel end positions (the two wound-up reel centres)
    use_custom_ends: BoolProperty(
        name="Custom Reel End Positions", default=False,
        description="Wound reel: put the two reel centres (supply core and take-up core) at the "
                    "positions below instead of at the 3D cursor")
    end_a: FloatVectorProperty(
        name="End A (Supply)", size=3, subtype='TRANSLATION', unit='LENGTH', default=(0.0, 0.0, 0.0),
        update=_end_a_edited, description="World position of the supply reel centre (frame 1 at the core)")
    end_b: FloatVectorProperty(
        name="End B (Take-up)", size=3, subtype='TRANSLATION', unit='LENGTH', default=(0.3, 0.0, 0.0),
        update=_end_b_edited, description="World position of the take-up reel centre")
    end_obj_a: PointerProperty(
        name="Object A", type=bpy.types.Object,
        description="Set by 'Set Reel End Positions' from the selection. While set, Build uses this "
                    "object's current origin for End A")
    end_obj_b: PointerProperty(
        name="Object B", type=bpy.types.Object,
        description="Set by 'Set Reel End Positions' from the selection. While set, Build uses this "
                    "object's current origin for End B")
    scale_multiplier: FloatProperty(
        name="Scale", default=1.0, min=0.01, max=1000.0, precision=3,
        description="Size multiplier on top of true scale. 1.0 = real size (35 mm film is "
                    "0.035 m wide, following the scene's Unit Scale). Use e.g. 10 for a "
                    "film strip ten times larger than life")


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def _set(node, names, value):
    if isinstance(names, str):
        names = (names,)
    for n in names:
        s = node.inputs.get(n)
        if s is not None:
            try:
                s.default_value = value
            except Exception:
                pass
            return True
    return False


def _new_material(name):
    mat = bpy.data.materials.new(name)
    mat[MARK] = True
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def get_holder_material():
    mat = bpy.data.materials.get("M_FilmHolder")
    if mat is not None and mat.node_tree is not None:
        return mat
    mat, nt = _new_material("M_FilmHolder")
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (320, 0)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    b.location = (0, 0)
    _set(b, 'Base Color', (0.0, 0.0, 0.0, 1.0))
    _set(b, 'Metallic', 0.0)
    _set(b, 'Roughness', 0.15)
    _set(b, 'IOR', 1.5)
    _set(b, 'Alpha', 1.0)
    _set(b, 'Thin Wall', True)
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return mat


def make_image_material(name, image=None, mapping=None, video_frame=None):
    """M_FilmImage: image -> Base Color, Roughness 0, Transmission 1, Thin Wall.

    mapping = from picture_mapping(): adds a Mapping node (and, for a cropped picture
    smaller than the frame window, a mask) so the rest of the window renders black.
    video_frame = 0-based frame of a movie (frozen via frame offset).
    """
    mat, nt = _new_material(name)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (700, 0)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    b.location = (360, 0)
    _set(b, 'Metallic', 0.0)
    _set(b, 'Roughness', 0.0)
    _set(b, 'IOR', 1.5)
    _set(b, 'Alpha', 1.0)
    _set(b, ('Transmission Weight', 'Transmission'), 1.0)
    _set(b, 'Thin Wall', True)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.location = (0, 0)
    tex.interpolation = 'Linear'
    tex.projection = 'FLAT'
    tex.extension = 'CLIP' if (mapping is not None and mapping[4]) else 'REPEAT'
    if image is not None:
        tex.image = image
    if video_frame is not None:
        iu = tex.image_user
        iu.frame_start = 1
        iu.frame_duration = 1
        iu.frame_offset = int(video_frame)
        iu.use_cyclic = False
        iu.use_auto_refresh = True
    color_out = tex.outputs['Color']
    if mapping is not None:
        uv = nt.nodes.new('ShaderNodeUVMap')
        uv.location = (-520, -20)
        mp = nt.nodes.new('ShaderNodeMapping')
        mp.location = (-260, -20)
        mp.vector_type = 'POINT'
        mp.inputs['Location'].default_value = (mapping[2], mapping[3], 0.0)
        mp.inputs['Scale'].default_value = (mapping[0], mapping[1], 1.0)
        nt.links.new(uv.outputs['UV'], mp.inputs['Vector'])
        nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
        if mapping[5] is not None:
            # 1 inside the picture rectangle, 0 outside: floor(r) == 0 on both axes
            mr = nt.nodes.new('ShaderNodeMapping')
            mr.location = (-260, -320)
            mr.vector_type = 'POINT'
            mr.inputs['Location'].default_value = (mapping[5][2], mapping[5][3], 0.0)
            mr.inputs['Scale'].default_value = (mapping[5][0], mapping[5][1], 1.0)
            nt.links.new(uv.outputs['UV'], mr.inputs['Vector'])
            fl = nt.nodes.new('ShaderNodeVectorMath')
            fl.operation = 'FLOOR'
            fl.location = (0, -320)
            nt.links.new(mr.outputs['Vector'], fl.inputs[0])
            ln = nt.nodes.new('ShaderNodeVectorMath')
            ln.operation = 'LENGTH'
            ln.location = (180, -320)
            nt.links.new(fl.outputs['Vector'], ln.inputs[0])
            inside = nt.nodes.new('ShaderNodeMath')
            inside.operation = 'LESS_THAN'
            inside.location = (360, -320)
            inside.inputs[1].default_value = 0.5
            nt.links.new(ln.outputs['Value'], inside.inputs[0])
            sc = nt.nodes.new('ShaderNodeVectorMath')
            sc.operation = 'SCALE'
            sc.location = (180, 0)
            nt.links.new(tex.outputs['Color'], sc.inputs[0])
            nt.links.new(inside.outputs['Value'], sc.inputs['Scale'])
            color_out = sc.outputs['Vector']
    nt.links.new(color_out, b.inputs['Base Color'])
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return mat


# ----------------------------------------------------------------------------
# Unwind along a path: per-reel settings, live mesh update, handlers
# ----------------------------------------------------------------------------
_MESH_CACHE = []
_SIG = {}
_BUSY = [False]
_BUILDING = [False]                 # True while a reel is being built / replaced
_FOLLOW = {"reel": None}            # the reel whose settings the panel last loaded
_CFG_KEY = "film_reel_cfg"          # custom property on a reel's strip: the settings it was built with
_CFG_SKIP = {"rna_type", "build_mode", "files", "end_obj_a", "end_obj_b"}
_MSGBUS_OWNER = object()


def snapshot_settings(fr):
    """The panel settings as JSON, stored on a reel's strip when it is built."""
    d = {}
    for prop in fr.bl_rna.properties:
        pid = prop.identifier
        if pid in _CFG_SKIP or prop.type in {'POINTER', 'COLLECTION'}:
            continue
        v = getattr(fr, pid)
        if prop.type == 'ENUM' and prop.is_enum_flag:
            v = sorted(v)
        elif getattr(prop, "is_array", False) or getattr(prop, "array_length", 0) > 0:
            v = list(v)
        d[pid] = v
    d["_files"] = [f.name for f in fr.files]
    d["_obj_a"] = fr.end_obj_a.name if fr.end_obj_a else ""
    d["_obj_b"] = fr.end_obj_b.name if fr.end_obj_b else ""
    return json.dumps(d)


def apply_settings(fr, d):
    """Copy a saved settings dict into the panel (without triggering the edit callbacks)."""
    _LOADING[0] = True
    try:
        for pid, v in d.items():
            if pid.startswith("_"):
                continue
            prop = fr.bl_rna.properties.get(pid)
            if prop is None or pid in _CFG_SKIP or prop.type in {'POINTER', 'COLLECTION'}:
                continue
            try:
                if prop.type == 'ENUM' and prop.is_enum_flag:
                    v = set(v)
                setattr(fr, pid, v)
            except Exception:
                pass
        if "_files" in d:
            fr.files.clear()
            for name in d["_files"]:
                fr.files.add().name = name
        fr.end_obj_a = bpy.data.objects.get(d.get("_obj_a", "")) if d.get("_obj_a") else None
        fr.end_obj_b = bpy.data.objects.get(d.get("_obj_b", "")) if d.get("_obj_b") else None
    finally:
        _LOADING[0] = False


def reel_settings_dict(strip):
    """What a reel is set to now: its build settings plus the live edits made since."""
    raw = strip.get(_CFG_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except Exception:
        return None
    uw = strip.film_reel_unwind
    d["layout_mode"] = 'SPIRAL' if uw.built_spiral else 'STRAIGHT'
    if uw.built_spiral:
        d["core_radius_mm"] = uw.supply_core_mm
        d["takeup_core_mm"] = uw.takeup_core_mm
        d["wind_spacing_mm"] = uw.wind_mm
    return d


def _redraw_all():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


def follow_selected_reel():
    """When another film reel made by this add-on becomes the active one, show ITS settings."""
    if _BUILDING[0] or _LOADING[0]:
        return
    ctx = bpy.context
    scene, vl = getattr(ctx, "scene", None), getattr(ctx, "view_layer", None)
    if scene is None or vl is None:
        return
    ob = vl.objects.active
    if ob is None:
        return
    reels = reel_collections(scene)
    col = next((c for c in ob.users_collection if c in reels), None)
    if col is None or _FOLLOW["reel"] == col.name:
        return
    _FOLLOW["reel"] = col.name
    strip = reel_strip(col)
    d = reel_settings_dict(strip) if strip is not None else None
    if d is None:                       # a reel built by an older version: nothing saved to show
        return
    apply_settings(scene.film_reel, d)
    _redraw_all()


def _on_active_changed():
    try:
        follow_selected_reel()
    except Exception as e:
        print("Film Reel follow selection:", e)


@bpy.app.handlers.persistent
def _follow_depsgraph_handler(scene, depsgraph=None):
    _on_active_changed()


@bpy.app.handlers.persistent
def _follow_load_post(*args):
    _FOLLOW["reel"] = None
    for o in bpy.data.objects:
        try:
            if o.get(MARK) and o.type == 'MESH' and o.film_reel_unwind.takeup_core_prev_mm < 0.0:
                o.film_reel_unwind.takeup_core_prev_mm = o.film_reel_unwind.takeup_core_mm
        except Exception:
            pass
    _subscribe_active()


def _subscribe_active():
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
        bpy.msgbus.subscribe_rna(key=(bpy.types.LayerObjects, "active"), owner=_MSGBUS_OWNER,
                                 args=(), notify=_on_active_changed)
    except Exception as e:
        print("Film Reel msgbus:", e)
HELPER = "film_reel_helper"


def _decoded():
    if not _MESH_CACHE:
        _MESH_CACHE.append(decode_mesh())
    return _MESH_CACHE[0]


def reel_strip(col):
    """The strip object of a reel collection."""
    if col is None:
        return None
    return next((o for o in col.objects if o.get(MARK) and o.type == 'MESH'), None)


def apply_frame_order(ob):
    """Point every frame's picture polygons at the right image material.

    Normal order: frame k shows picture k. With Reverse Film on, frame k shows picture
    n-1-k. Only the polygons' material slots change, so the geometry (and any unwinding)
    is untouched.
    """
    uw = ob.film_reel_unwind
    mesh = ob.data
    n = uw.n_frames
    npoly = len(mesh.polygons)
    if ob.type != 'MESH' or n <= 0 or npoly == 0 or npoly % n:
        return
    base = np.asarray(_decoded()[2], dtype=np.int64)
    nf = npoly // n
    if len(base) != nf:
        return
    is_img = base == IMAGE_SLOT
    k = np.arange(n, dtype=np.int64)
    src = (n - 1 - k) if uw.reverse_film else k
    idx = np.where(is_img[None, :], src[:, None] + 1, 0).ravel()
    mesh.polygons.foreach_set("material_index", idx.astype(np.int32))
    mesh.update()


def _reverse_film_changed(self, context):
    try:
        apply_frame_order(self.id_data)
    except Exception as e:  # never break the UI over a refresh
        print("Film Reel reverse:", e)


def _rebend_reel(ob, uw):
    """Re-bend the strip after a live change of cores or wind spacing."""
    if not uw.built_spiral or uw.n_frames <= 0:
        return
    _SIG.pop(ob.name, None)
    if uw.enabled and uw.path is not None and uw.takeup is not None:
        update_unwind_reel(ob)
    else:
        restore_wound_reel(ob)


def _wind_changed(self, context):
    """Live edit of the wind spacing: re-bend the reel straight away."""
    try:
        ob = self.id_data
        try:
            context.scene.film_reel.wind_spacing_mm = self.wind_mm        # next build starts here
        except Exception:
            pass
        new = max(self.wind_mm, WIND_MIN_MM) * self.units_per_mm
        if abs(new - self.wind) < 1e-12:
            return
        self.wind = new
        _rebend_reel(ob, self)
    except Exception as e:
        print("Film Reel wind spacing:", e)


def _reel_full_radius(uw, core_r):
    """Outer radius (Blender units) of this reel's film wound on a core of radius core_r."""
    pitch = _frame_pitch(_decoded()[0], uw.film_scale)
    return _full_radius(core_r, uw.wind, uw.n_frames * pitch, pitch)


def shift_path_end(ob, end, d_radius):
    """Keep the film path just outside a reel when that reel's size changes.

    Moves the FIRST ('A', supply reel) or LAST ('B', take-up reel) point of the reel's path curve
    straight away from / towards that reel's centre by d_radius, within the reel's plane. The
    point's gap to the surface of the roll therefore stays exactly what it was. Bezier handles
    travel with their point. Returns True if a point was moved."""
    uw = ob.film_reel_unwind
    pob, tk = uw.path, uw.takeup
    if abs(d_radius) < 1e-12 or pob is None or pob.type != 'CURVE' or not pob.data.splines:
        return False
    if (end == 'B' and tk is None) or pob.mode == 'EDIT':
        return False
    M = np.array(ob.matrix_world)
    Minv = np.linalg.inv(M)
    Pm = np.array(pob.matrix_world)
    P3inv = np.linalg.inv(Pm[:3, :3])
    if end == 'A':
        c = np.zeros(2)
    else:
        c = (Minv @ np.append(np.array(tk.matrix_world.translation[:]), 1.0))[1:3]
    sp = pob.data.splines[0]                      # the spline the film actually follows
    pts = sp.bezier_points if sp.type == 'BEZIER' else sp.points
    if len(pts) == 0:
        return False
    pt = pts[0] if end == 'A' else pts[len(pts) - 1]
    co = np.array(pt.co[:3])
    w = Pm[:3, :3] @ co + Pm[:3, 3]
    q = Minv[:3, :3] @ w + Minv[:3, 3]
    v = q[1:3] - c
    d = float(np.hypot(v[0], v[1]))
    if d < 1e-12:
        return False
    q[1:3] = c + v / d * max(d + d_radius, 0.0)
    w2 = M[:3, :3] @ q + M[:3, 3]
    delta = P3inv @ (w2 - w)
    if sp.type == 'BEZIER':
        pt.co = tuple(float(x) for x in co + delta)
        pt.handle_left = tuple(float(x) for x in np.array(pt.handle_left[:]) + delta)
        pt.handle_right = tuple(float(x) for x in np.array(pt.handle_right[:]) + delta)
    else:
        pt.co = (pt.co[0] + float(delta[0]), pt.co[1] + float(delta[1]),
                 pt.co[2] + float(delta[2]), pt.co[3])
    pob.data.update_tag()
    return True


def _supply_core_changed(self, context):
    """Live edit of End A's (supply reel) core radius: re-bend the reel straight away, and carry
    the start of the path along so it stays just outside the roll."""
    try:
        ob = self.id_data
        try:
            context.scene.film_reel.core_radius_mm = self.supply_core_mm   # next build starts here
        except Exception:
            pass
        new = self.supply_core_mm * self.units_per_mm
        if abs(new - self.core_r) < 1e-12:
            return
        follow = self.built_spiral and self.n_frames > 0
        old_full = _reel_full_radius(self, self.core_r) if follow else 0.0
        self.core_r = new
        if follow:
            try:
                shift_path_end(ob, 'A', _reel_full_radius(self, new) - old_full)
            except Exception as e:
                print("Film Reel path follow:", e)
        _rebend_reel(ob, self)
    except Exception as e:  # never break the UI over a refresh
        print("Film Reel core radius:", e)


def _takeup_core_changed(self, context):
    try:
        context.scene.film_reel.takeup_core_mm = self.takeup_core_mm       # next build starts here
    except Exception:
        pass
    try:
        prev = self.takeup_core_prev_mm
        if prev > 0.0 and abs(self.takeup_core_mm - prev) > 1e-9 and self.built_spiral and self.n_frames > 0:
            old_full = _reel_full_radius(self, prev * self.units_per_mm)
            new_full = _reel_full_radius(self, self.takeup_core_mm * self.units_per_mm)
            shift_path_end(self.id_data, 'B', new_full - old_full)
        self.takeup_core_prev_mm = self.takeup_core_mm
    except Exception as e:
        print("Film Reel path follow:", e)
    _unwind_params_changed(self, context)


def _unwind_params_changed(self, context):
    ob = self.id_data
    _SIG.pop(ob.name, None)
    try:
        update_unwind_reel(ob)
    except Exception as e:  # never break the UI over a refresh
        print("Film Reel unwind:", e)


class FilmReelUnwind(PropertyGroup):
    # --- remembered from the build, needed to re-bend the strip
    built_spiral: BoolProperty(default=False)
    n_frames: IntProperty(default=0)
    film_scale: FloatProperty(default=1.0, precision=9)
    core_r: FloatProperty(default=0.0, precision=9)
    wind: FloatProperty(default=0.0, precision=9)
    units_per_mm: FloatProperty(default=0.001, precision=9)
    status: StringProperty()
    total_length: FloatProperty(default=0.0, precision=9)   # film length fed from progress 0 -> 1
    # --- user settings
    enabled: BoolProperty(default=False)
    progress: FloatProperty(
        name="Unwind", default=0.0, min=0.0, max=1.0, precision=6, step=1,
        update=_unwind_params_changed,
        description="0 = all film on the supply reel, 1 = all of it re-wound on the take-up reel. "
                    "Type any value (as many decimals as you like) or drag. "
                    "Animate this value (I over the field) to play the unwind")
    arrow_step: FloatProperty(
        name="Arrow Step", default=0.01, min=0.0, max=1.0, precision=6, step=1,
        description="How far the arrows beside the Unwind value move it per click, in the same "
                    "units as Unwind (0 = all film on the supply reel, 1 = all on the take-up). "
                    "Any value, with as many decimals as you like")
    path: PointerProperty(
        name="Path", type=bpy.types.Object, update=_unwind_params_changed,
        poll=lambda self, ob: ob.type == 'CURVE',
        description="Curve the free film follows from the supply reel to the take-up reel. "
                    "It is flattened onto the reel's plane (the plane across the reel axis)")
    takeup: PointerProperty(
        name="Take-up Reel", type=bpy.types.Object, update=_unwind_params_changed,
        description="Object marking the centre of the take-up reel (an Empty works well). "
                    "Its position is used, flattened onto the reel's plane")
    wind_mm: FloatProperty(
        name="Wind Spacing", default=1.0, min=0.2, soft_max=10.0, update=_wind_changed,
        description="Radial growth of the spiral per turn, in mm of film (real film is ~0.14 mm; "
                    "larger looks better). Kept above a minimum so the turns never pile up")
    supply_core_mm: FloatProperty(
        name="End A Core Radius", default=60.0, min=1.0, soft_max=300.0,
        update=_supply_core_changed,
        description="Radius of the core of the supply reel (End A), in mm of film. "
                    "Frame 1 sits on this core")
    takeup_core_prev_mm: FloatProperty(default=-1.0, precision=6)   # End B core before the last change
    takeup_core_mm: FloatProperty(
        name="End B Core Radius", default=60.0, min=1.0, soft_max=300.0,
        update=_takeup_core_changed,
        description="Radius of the core of the take-up reel (End B), in mm of film")
    path_segments: IntProperty(
        name="Path Subdivisions", default=1, min=1, max=64,
        description="Number of segments (axes) the film path is divided into. 1 = a single "
                    "straight run; every extra segment adds a control point you can move to bend the path")
    align_frame: IntProperty(
        name="Frame", default=1, min=1, max=100000,
        description="Reel frame number (the same numbering as From / To) whose centre is moved "
                    "to the chosen object when you click Align Frame to Object")
    align_object: PointerProperty(
        name="Object", type=bpy.types.Object,
        description="The object whose origin the chosen frame's centre is unwound to")
    reverse_film: BoolProperty(
        name="Reverse Film", default=False, update=_reverse_film_changed,
        description="Reverse the order of the image frames on the film "
                    "(the last picture comes first)")


def _curve_world_points(ob, depsgraph=None):
    """First spline of a curve object, as world-space points (m x 3)."""
    local = _curve_local_points(ob, depsgraph)
    if not len(local):
        return np.zeros((0, 3))
    M = np.array(ob.matrix_world)
    return local @ M[:3, :3].T + M[:3, 3]


def _curve_local_points(ob, depsgraph=None):
    """First spline of a curve object, as points in the curve's own space (m x 3)."""
    cu = ob.data
    if not cu.splines:
        return np.zeros((0, 3))
    sp = cu.splines[0]
    pts = []
    if sp.type == 'BEZIER':
        bp = sp.bezier_points
        cnt = len(bp)
        segs = cnt if (sp.use_cyclic_u and cnt > 2) else cnt - 1
        t = np.linspace(0.0, 1.0, 32, endpoint=False)[:, None]
        for i in range(max(segs, 0)):
            a, c = bp[i], bp[(i + 1) % cnt]
            p0, p1, p2, p3 = (np.array(v[:]) for v in (a.co, a.handle_right, c.handle_left, c.co))
            pts.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3)
        if cnt:
            pts.append(np.array(bp[0 if (sp.use_cyclic_u and cnt > 2) else cnt - 1].co[:])[None, :])
        local = np.vstack(pts) if pts else np.zeros((0, 3))
    elif sp.type == 'POLY':
        local = np.array([p.co[:3] for p in sp.points], dtype=np.float64).reshape(-1, 3)
        if sp.use_cyclic_u and len(local) > 2:
            local = np.vstack([local, local[:1]])
    else:   # NURBS etc: use the evaluated curve
        dg = depsgraph or bpy.context.evaluated_depsgraph_get()
        eob = ob.evaluated_get(dg)
        me = eob.to_mesh()
        try:
            local = np.array([v.co[:] for v in me.vertices], dtype=np.float64).reshape(-1, 3)
        finally:
            eob.to_mesh_clear()
    return local


def resample_polyline(pts, segments):
    """Evenly (by arc length) resample a polyline into `segments` segments -> segments+1 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    segments = max(1, int(segments))
    if len(pts) < 2:
        return np.repeat(pts[:1], segments + 1, axis=0) if len(pts) else np.zeros((segments + 1, 3))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    if cum[-1] < 1e-12:
        return np.repeat(pts[:1], segments + 1, axis=0)
    t = np.linspace(0.0, cum[-1], segments + 1)
    return np.stack([np.interp(t, cum, pts[:, k]) for k in range(3)], axis=1)


def set_path_points(curve_ob, pts, smooth):
    """Replace the curve's splines with ONE bezier spline through pts (curve-local coords)."""
    cu = curve_ob.data
    cu.splines.clear()
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    htype = 'AUTO' if smooth else 'VECTOR'
    for bp, co in zip(sp.bezier_points, pts):
        bp.co = tuple(float(c) for c in co)
        bp.handle_left_type = bp.handle_right_type = htype
    sp.use_cyclic_u = False
    for bp in sp.bezier_points:          # second pass: handles depend on neighbours, so
        bp.handle_left_type = bp.handle_right_type = htype   # force a recalculation now


def _apply_to_mesh(mesh, pos, vnorm=None, lnorm=None):
    mesh.vertices.foreach_set("co", np.ascontiguousarray(pos, dtype=np.float32).ravel())
    mesh.update()
    try:
        if lnorm is None:
            loop_v = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", loop_v)
            lnorm = vnorm[loop_v]
        mesh.normals_split_custom_set(np.asarray(lnorm).tolist())
    except Exception:
        pass
    mesh.update()


def update_unwind_reel(ob, depsgraph=None):
    """Re-bend a reel's strip for its current unwind progress."""
    if _BUSY[0] or ob is None or ob.type != 'MESH' or not ob.get(MARK):
        return
    uw = ob.film_reel_unwind
    if not uw.enabled or not uw.built_spiral or uw.path is None or uw.takeup is None:
        return
    mesh = ob.data
    verts = _decoded()[0]
    if uw.n_frames * len(verts) != len(mesh.vertices):
        return
    Minv = np.array(ob.matrix_world.inverted())

    def to_local(P):                     # world points (m x 3) -> reel plane (m x 2: local Y, Z)
        q = P @ Minv[:3, :3].T + Minv[:3, 3]
        return q[:, 1:3]

    world = _curve_world_points(uw.path, depsgraph)
    if len(world) < 2:
        return
    path = to_local(world)
    c_t = to_local(np.array([uw.takeup.matrix_world.translation[:]]))[0]
    sig = (float(uw.progress), path.tobytes(), c_t.tobytes(), uw.takeup_core_mm, uw.core_r, uw.wind,
           uw.n_frames)
    if _SIG.get(ob.name) == sig:
        return
    _BUSY[0] = True
    try:
        pos, vnorm, warn, total = unwind_geometry(
            verts, uw.film_scale, uw.n_frames, uw.core_r, uw.wind,
            uw.takeup_core_mm * uw.units_per_mm, path, c_t, uw.progress)
        _apply_to_mesh(mesh, pos, vnorm=vnorm)
        _SIG[ob.name] = sig
        if uw.status != warn:
            uw.status = warn
        if abs(uw.total_length - total) > 1e-6:
            uw.total_length = total
    finally:
        _BUSY[0] = False


def restore_wound_reel(ob):
    """Put the strip back into its plain wound-reel shape."""
    uw = ob.film_reel_unwind
    verts, faces, mats, uvs = _decoded()
    pos, _f, lnorm, _m, _u = strip_geometry(verts, faces, mats, uvs, uw.film_scale, uw.n_frames,
                                            'SPIRAL', uw.core_r, uw.wind)
    _apply_to_mesh(ob.data, pos, lnorm=lnorm)
    _SIG.pop(ob.name, None)


@bpy.app.handlers.persistent
def _unwind_frame_handler(scene, depsgraph=None):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.get(MARK):
            try:
                update_unwind_reel(ob, depsgraph)
            except Exception as e:
                print("Film Reel unwind:", e)


@bpy.app.handlers.persistent
def _unwind_depsgraph_handler(scene, depsgraph=None):
    """Keep every unwinding reel in step when the reel, its path or its take-up is moved,
    rotated or reshaped (the signature check makes this a no-op when nothing relevant changed)."""
    if _BUSY[0]:
        return
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.get(MARK) and ob.film_reel_unwind.enabled:
            try:
                update_unwind_reel(ob, depsgraph)
            except Exception as e:
                print("Film Reel unwind:", e)


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def build_strip_mesh(scale, n, mode, r0, wind):
    verts, faces, mats, uvs = decode_mesh()
    pos, all_faces, lnorm, mat_idx, uv_all = strip_geometry(verts, faces, mats, uvs, scale,
                                                           n, mode, r0, wind)
    mesh = bpy.data.meshes.new("Film Reel Strip")
    mesh[MARK] = True
    mesh.from_pydata(pos.tolist(), [], all_faces)
    mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", uv_all)
    mesh.polygons.foreach_set("material_index", mat_idx.tolist())
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    try:  # the source model has all normals pointing along the film's face normal
        mesh.normals_split_custom_set(lnorm.tolist())
    except Exception:
        pass
    return mesh


def _walk_collections(col):
    for c in col.children:
        yield c
        yield from _walk_collections(c)


def reel_collections(scene):
    """Every film reel collection made by this add-on that lives in the scene."""
    return [c for c in _walk_collections(scene.collection) if c.get(MARK)]


def reel_serial(col):
    return int(col.get(SERIAL, 0))


def target_reel(context):
    """The reel 'Replace' acts on: the one holding the active object, else the newest."""
    reels = reel_collections(context.scene)
    if not reels:
        return None
    ob = context.view_layer.objects.active
    if ob is not None:
        for c in ob.users_collection:
            if c in reels:
                return c
    return max(reels, key=reel_serial)


def remove_reel(col, keep=(), keep_objects=()):
    """Delete a reel collection and its orphaned datablocks. `keep` lists datablocks that must
    survive even though nothing uses them yet (the new reel's mesh while it is being built -
    without this the clean-up below deleted it and Replace Reel failed with 'StructRNA of type
    Mesh has been removed')."""
    keep_ptr = {d.as_pointer() for d in keep if d is not None}
    protect = {o.as_pointer() for o in keep_objects if o is not None}
    for ob in list(col.all_objects):
        if ob.as_pointer() in protect:      # the path / take-up: they outlive a replaced reel
            doomed = {col} | set(col.children_recursive)
            for c in list(ob.users_collection):
                if c in doomed:
                    c.objects.unlink(ob)
            if not ob.users_collection:
                bpy.context.scene.collection.objects.link(ob)
            continue
        bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.collections.remove(col)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for d in list(coll):
            if d.get(MARK) and d.users == 0 and d.as_pointer() not in keep_ptr:
                coll.remove(d)


def resolve_end_positions(fr):
    """World positions (End A, End B) as mathutils Vectors. A picked object wins over typed
    coordinates and is read at call time, so Build always uses where the object is now."""
    from mathutils import Vector
    a = fr.end_obj_a.matrix_world.translation.copy() if fr.end_obj_a else Vector(fr.end_a)
    b = fr.end_obj_b.matrix_world.translation.copy() if fr.end_obj_b else Vector(fr.end_b)
    return a, b


def ends_rotation_matrix(a, b):
    """3x3 (numpy) whose columns are the reel's local X (axis), Y, Z in world space, chosen so the
    line a -> b lies in the reel plane and runs along local -Y (where the take-up sits).
    The axis is horizontal where possible, so two points at the same height give an upright reel."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    d = d / np.linalg.norm(d)
    axis = np.cross([0.0, 0.0, 1.0], d)
    if np.linalg.norm(axis) < 1e-6:                       # the ends are stacked vertically
        axis = np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    y = -d
    z = np.cross(axis, y)
    return np.stack([axis, y, z], axis=1)


def _full_radius(core_r, wind, length, pitch):
    b = wind / (2.0 * math.pi)
    arc, th = spiral_table(core_r, b, length + pitch)
    return core_r + b * float(np.interp(length, arc, th))


def _reel_collection(ob, scene):
    """The collection holding a reel's strip (where its path and take-up belong)."""
    return ob.users_collection[0] if ob.users_collection else scene.collection


def _move_into_reel_collection(h, ob):
    """Put a path / take-up made by this add-on into its reel's collection."""
    if not h.get(HELPER) or not ob.users_collection:
        return
    tgt = ob.users_collection[0]
    if tgt not in h.users_collection:
        tgt.objects.link(h)
    root = bpy.context.scene.collection
    if root != tgt and root in h.users_collection:
        root.objects.unlink(h)


def create_unwind_helpers(context, ob, takeup_dist=None):
    """Create the take-up marker and path curve of a wound reel when they are missing, and make
    both children of the reel so moving / rotating the reel carries them along as ONE unit
    (they used to be loose, so a moved reel no longer matched its path and the film folded)."""
    from mathutils import Matrix, Vector
    uw = ob.film_reel_unwind
    scene = context.scene
    context.view_layer.update()
    M = ob.matrix_world.copy()
    ys = [v[1] for v in _decoded()[0]]
    pitch = (max(ys) - min(ys)) * uw.film_scale
    L = uw.n_frames * pitch
    r_full = _full_radius(uw.core_r, uw.wind, L, pitch)
    r0t = uw.takeup_core_mm * uw.units_per_mm
    r_t_full = _full_radius(r0t, uw.wind, L, pitch)
    dist = takeup_dist if takeup_dist else 4.0 * r_full

    if uw.takeup is None:
        em = bpy.data.objects.new("Film Reel Take-up", None)
        em.empty_display_type = 'CIRCLE'
        em.empty_display_size = r0t
        em[HELPER] = True
        _reel_collection(ob, scene).objects.link(em)
        em.matrix_world = M @ Matrix.Translation((0.0, -dist, 0.0)) @ Matrix.Rotation(math.radians(90.0), 4, 'Y')
        uw.takeup = em
    if uw.path is None:
        cu = bpy.data.curves.new("Film Reel Path", 'CURVE')
        cu.dimensions = '3D'
        pob = bpy.data.objects.new("Film Reel Path", cu)
        pob[HELPER] = True
        _reel_collection(ob, scene).objects.link(pob)
        n = max(1, int(uw.path_segments))
        a = np.array([0.0, 0.0, r_full])
        b = np.array([0.0, -dist, r_t_full])
        pts = [M @ Vector(tuple(a + (b - a) * (i / n))) for i in range(n + 1)]
        set_path_points(pob, [tuple(v) for v in pts], smooth=False)
        uw.path = pob
    adopt_helpers(ob)
    context.view_layer.update()


def adopt_helpers(ob):
    """Parent the reel's path and take-up to the reel, keeping where they are in the world."""
    uw = ob.film_reel_unwind
    # A just-built reel still has a stale (identity) world matrix until the scene is evaluated.
    # Parenting with that stale matrix left the path and take-up displaced / rotated against the
    # reel's plane after Replace Reel, so the film could no longer travel along the path.
    bpy.context.view_layer.update()
    for h in (uw.path, uw.takeup):
        if h is not None and h != ob:
            _move_into_reel_collection(h, ob)
        if h is None or h.parent == ob or h == ob:
            continue
        mw = h.matrix_world.copy()
        h.parent = ob
        h.matrix_parent_inverse = ob.matrix_world.inverted()
        h.matrix_world = mw


def build_reel(context, fr, replace_col=None):
    _BUILDING[0] = True
    try:
        return _build_reel(context, fr, replace_col)
    finally:
        _BUILDING[0] = False


def make_frame_materials(context, source_type, directory, names, idx_map, limit, pack, mapping,
                         serial, first_frame, progress=True):
    """Load the footage frames and make one M_FilmImage material per reel frame.

    idx_map holds the source frame index for every reel frame. Returns (materials, images)
    where images maps source index -> image datablock (one per distinct source frame).
    """
    images, mats = {}, []
    wm = context.window_manager
    if progress:
        wm.progress_begin(0, max(1, len(idx_map)))
    try:
        for k, src in enumerate(idx_map):
            if src not in images:
                if source_type == 'SEQUENCE':
                    img = bpy.data.images.load(os.path.join(directory, names[src]),
                                               check_existing=False)
                    img[MARK] = True
                    w, h = img.size
                    if limit and max(w, h) > limit:
                        kk = limit / float(max(w, h))
                        img.scale(max(1, int(round(w * kk))), max(1, int(round(h * kk))))
                        if pack:
                            try:
                                img.pack(as_png=True)
                            except Exception:
                                pass
                else:
                    img = bpy.data.images.load(directory, check_existing=False)
                    img[MARK] = True
                images[src] = img
            img = images[src]
            mats.append(make_image_material("M_FilmImage_R%d_%04d" % (serial, first_frame + k),
                                            img, mapping,
                                            video_frame=src if source_type == 'VIDEO' else None))
            if progress:
                wm.progress_update(k)
    finally:
        if progress:
            wm.progress_end()
    return mats, images


def _build_reel(context, fr, replace_col=None):
    """Build a reel. With replace_col, that earlier reel is swapped out (after the
    new one has been built successfully, so a failure never loses the old reel).

    The replacement is a FULL RESET: the new reel gets the position, cores, wind spacing,
    unwind set-up, frame order and flips that the panel is currently set to, not those of
    the reel it replaces. Only the old reel's collection / object names are reused."""
    scene = context.scene
    upm = units_per_mm(scene, fr)              # Blender units per real mm of film
    scale = FILM_WIDTH[fr.film_size] / MESH_WIDTH * upm
    idx_map = source_index_map(fr)
    n = len(idx_map)
    first_frame = effective_range(fr)[0]
    serial = max([reel_serial(c) for c in bpy.data.collections if c.get(MARK)] + [0]) + 1

    # --- images (one datablock per distinct source frame) and one material per frame
    names = [f.name for f in fr.files]
    mats, images = make_frame_materials(
        context, fr.source_type, fr.directory, names, idx_map, resolved_limit(fr), fr.pack_images,
        picture_mapping(fr.src_width, fr.src_height, picture_aspect(fr), fr.image_fit),
        serial, first_frame)

    # --- one continuous strip mesh: frame edges touch, all frames in a single object
    # (Replace Reel is a full reset: everything below comes from the current panel settings)
    core_mm = fr.core_radius_mm
    wind_mm = max(fr.wind_spacing_mm, WIND_MIN_MM)
    mesh = build_strip_mesh(scale, n, fr.layout_mode,
                            core_mm * upm,
                            wind_mm * upm)
    mesh.materials.append(get_holder_material())       # slot 0: base + perforations
    for m in mats:                                       # slot k+1: picture of frame k
        mesh.materials.append(m)

    # --- custom reel end positions (wound reel only)
    ends = None
    if fr.use_custom_ends and fr.layout_mode == 'SPIRAL':
        ends = resolve_end_positions(fr)
        if (ends[1] - ends[0]).length < 1e-6:
            raise ValueError("Reel End A and End B are at the same position")

    # --- swap out the old reel only now that the new one is ready; keep its names
    col_name, ob_name = COLLECTION_NAME, ROOT_NAME
    if replace_col is not None:
        col_name = replace_col.name
        ob_name = next((o.name for o in replace_col.objects if o.get(MARK)), ROOT_NAME)
        old_strip = reel_strip(replace_col)
        keep_world = {}                          # path / take-up objects the USER supplied survive
        doomed_data = []                         # the add-on's own helpers are deleted with the reel
        if old_strip is not None:
            u = old_strip.film_reel_unwind
            for h in (u.path, u.takeup):
                if h is None:
                    continue
                if h.get(HELPER):
                    if h.data is not None:
                        doomed_data.append(h.data)
                else:
                    keep_world[h] = h.matrix_world.copy()
        remove_reel(replace_col, keep=[mesh] + list(mats) + list(images.values()),
                    keep_objects=list(keep_world))
        for h, mw in keep_world.items():
            h.parent = None
            h.matrix_world = mw
        for d in doomed_data:
            try:
                if d.users == 0 and isinstance(d, bpy.types.Curve):
                    bpy.data.curves.remove(d)
            except Exception:
                pass

    col = bpy.data.collections.new(col_name)
    col[MARK] = True
    col[SERIAL] = serial
    scene.collection.children.link(col)
    strip = bpy.data.objects.new(ob_name, mesh)
    strip[MARK] = True
    strip.location = scene.cursor.location
    if fr.layout_mode == 'STRAIGHT':
        strip.rotation_euler = (math.radians(90.0), 0.0, 0.0) if fr.stand_upright else (0.0, 0.0, 0.0)
    elif ends is not None:
        from mathutils import Matrix
        R = ends_rotation_matrix(ends[0], ends[1])
        strip.location = ends[0]
        strip.rotation_euler = Matrix(R.tolist()).to_euler()
    else:
        strip.rotation_euler = (0.0, 0.0, math.radians(90.0))
    col.objects.link(strip)

    # remember what the unwind feature needs to re-bend this strip later
    uw = strip.film_reel_unwind
    uw.built_spiral = fr.layout_mode == 'SPIRAL'
    uw.n_frames = n
    uw.film_scale = scale
    uw.core_r = core_mm * upm
    uw.wind = wind_mm * upm
    uw.units_per_mm = upm
    uw.wind_mm = wind_mm
    uw.supply_core_mm = core_mm                  # live End A core editor starts at the build value
    uw.takeup_core_mm = fr.takeup_core_mm
    uw.takeup_core_prev_mm = fr.takeup_core_mm   # lets later End B edits carry the path along
    if ends is not None:
        # fresh helpers: take-up marker exactly on End B, path between the two reels
        context.view_layer.update()
        create_unwind_helpers(context, strip, takeup_dist=(ends[1] - ends[0]).length)

    for ob in context.selected_objects:
        ob.select_set(False)
    strip.select_set(True)
    context.view_layer.objects.active = strip
    strip[_CFG_KEY] = snapshot_settings(fr)      # what this reel was built with (shown on selecting it)
    _FOLLOW["reel"] = col.name
    return n


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def _probe_image(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        frames = getattr(img, "frame_duration", 1) if img.source == 'MOVIE' else 1
    finally:
        bpy.data.images.remove(img)
    return w, h, frames


def probe_source(context, directory, names):
    """Read a picked video / image sequence. Returns a dict with the source type, the
    directory (the movie's full path for a video), the file names, size, frame count and fps.
    Raises ValueError when the footage can't be read."""
    is_video = (len(names) == 1 and os.path.splitext(names[0])[1].lower() in movie_extensions())
    if is_video:
        path = os.path.join(directory, names[0])
        w, h, nframes = _probe_image(path)
        fps = 0.0
        try:
            clip = bpy.data.movieclips.load(path)
            fps = float(getattr(clip, "fps", 0.0))
            nframes = clip.frame_duration or nframes
            bpy.data.movieclips.remove(clip)
        except Exception:
            pass
        kind, where, names = 'VIDEO', path, [names[0]]
    else:
        names = expand_sequence(directory, names)
        w, h, _ = _probe_image(os.path.join(directory, names[0]))
        nframes, fps = len(names), 0.0
        kind, where = 'SEQUENCE', directory
    if w <= 0 or h <= 0 or nframes <= 0:
        raise ValueError("Could not read image size / frame count")
    if fps <= 0:
        fps = float(context.scene.render.fps) / context.scene.render.fps_base
    return {"type": kind, "directory": where, "names": names, "w": w, "h": h,
            "frames": nframes, "fps": fps}


def _picked_names(op):
    """(directory, names) of the files picked in a file browser operator."""
    directory = bpy.path.abspath(op.directory)
    names = [f.name for f in op.files if f.name]
    if not names and op.filepath:
        directory, fn = os.path.split(bpy.path.abspath(op.filepath))
        names = [fn]
    return directory, names


class FILMREEL_OT_load(Operator):
    bl_idname = "film_reel.load_source"
    bl_label = "Load Main Video"
    bl_description = ("Pick an image sequence (select all frames, or just the first one) "
                      "or a video file")

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    filter_image: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_movie: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_folder: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        fr = context.scene.film_reel
        directory, names = _picked_names(self)
        if not names:
            self.report({'WARNING'}, "No file selected")
            return {'CANCELLED'}
        try:
            src = probe_source(context, directory, names)
        except Exception as e:
            self.report({'ERROR'}, "Could not read the source: %s" % e)
            return {'CANCELLED'}
        fr.source_type, fr.directory = src["type"], src["directory"]
        fr.files.clear()
        for nme in src["names"]:
            fr.files.add().name = nme
        fr.src_width, fr.src_height, fr.src_frames = src["w"], src["h"], src["frames"]
        fps = src["fps"]
        fr.source_fps = fps
        match = next((i for i, _, _ in FPS_ITEMS if i != 'CUSTOM' and abs(float(i) - fps) < 0.005), None)
        if match:
            fr.fps_preset = match
        else:
            fr.fps_preset, fr.custom_fps = 'CUSTOM', fps
        fr.frame_start = 1
        fr.frame_end = recommended_frames(fr)
        return {'FINISHED'}


def _purge_orphans():
    """Remove this add-on's materials / images that nothing uses any more."""
    for coll in (bpy.data.materials, bpy.data.images):
        for d in list(coll):
            if d.get(MARK) and d.users == 0:
                coll.remove(d)


class FILMREEL_OT_replace_source(Operator):
    bl_idname = "film_reel.replace_source"
    bl_label = "Replace Movie Clip / Image Sequence"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Pick a different video or image sequence and swap ONLY the pictures on the "
                      "selected reel. Its position, rotation, frame count, cores, unwinding, flips, "
                      "color space and all other settings stay exactly as they are")

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    filter_image: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_movie: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_folder: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        st = target_strip(context)
        return st is not None and st.film_reel_unwind.n_frames > 0

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        col = target_reel(context)
        strip = reel_strip(col)
        if strip is None:
            self.report({'ERROR'}, "No reel to update")
            return {'CANCELLED'}
        uw = strip.film_reel_unwind
        mesh = strip.data
        n = uw.n_frames
        if n <= 0 or len(mesh.materials) != n + 1:
            self.report({'ERROR'}, "This reel's frame materials don't match its frame count")
            return {'CANCELLED'}
        directory, names = _picked_names(self)
        if not names:
            self.report({'WARNING'}, "No file selected")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            src = probe_source(context, directory, names)
        except Exception as e:
            self.report({'ERROR'}, "Could not read the source: %s" % e)
            return {'CANCELLED'}

        # the reel's own settings (what it was built with + live edits); older reels fall back to the panel
        try:
            saved = json.loads(strip.get(_CFG_KEY) or "")
        except Exception:
            saved = json.loads(snapshot_settings(scene.film_reel))
        preset = saved.get("fps_preset", '24')
        reel_fps = max(0.001, float(saved.get("custom_fps", 24.0))) if preset == 'CUSTOM' else float(preset)

        sw, sh, sframes, sfps = src["w"], src["h"], src["frames"], src["fps"]
        first = max(1, int(saved.get("frame_start", 1)))      # the reel's frame range never moves
        note = ""
        ratio = sfps / reel_fps
        raw_idx = [int((first - 1 + j) * ratio + 1e-6) for j in range(n)]
        held = sum(1 for i in raw_idx if i > sframes - 1)
        idx_map = [min(sframes - 1, i) for i in raw_idx]

        look = SimpleNamespace(aspect_preset=saved.get("aspect_preset", 'RECOMMENDED'),
                               custom_aspect=saved.get("custom_aspect", 16.0 / 9.0),
                               src_width=sw, src_height=sh)
        mapping = picture_mapping(sw, sh, picture_aspect(look), saved.get("image_fit", 'STRETCH'))
        mode = saved.get("max_image_size", 'RECOMMENDED')
        if mode == 'RECOMMENDED':
            limit = recommended_limit(src["type"], sw, sh, len(set(idx_map)))
        else:
            limit = int(mode)

        flips = reel_flip_state(col)             # remembered before the old pictures go
        cs = reel_colorspace(col)
        serial = reel_serial(col) or 1

        new_mats, new_images = [], {}
        try:
            new_mats, new_images = make_frame_materials(
                context, src["type"], src["directory"], src["names"], idx_map, limit,
                bool(saved.get("pack_images", False)), mapping, serial, first)
        except Exception as e:                   # nothing on the reel has been touched yet
            for m in new_mats:
                bpy.data.materials.remove(m)
            _purge_orphans()
            self.report({'ERROR'}, "Could not load the new frames: %s" % e)
            return {'CANCELLED'}

        for k, m in enumerate(new_mats):         # slot k + 1 = picture of frame k (slot 0 = holder)
            mesh.materials[k + 1] = m
        for axis, flipped in zip('HV', flips):
            if flipped:
                for m in new_mats:
                    flip_material(m, axis)
        if cs and cs != "Mixed":
            for img in new_images.values():
                try:
                    img.colorspace_settings.name = cs
                except Exception:
                    pass
        _purge_orphans()                         # the old pictures
        for k, m in enumerate(new_mats):
            m.name = "M_FilmImage_R%d_%04d" % (serial, first + k)

        # remember the new footage on the reel (only the source fields; every other setting is kept)
        try:
            d = json.loads(strip.get(_CFG_KEY) or "")
        except Exception:
            d = saved
        d.update(source_type=src["type"], directory=src["directory"], _files=list(src["names"]),
                 src_width=sw, src_height=sh, src_frames=sframes, source_fps=sfps,
                 frame_start=first, frame_end=saved.get("frame_end", first + n - 1))
        strip[_CFG_KEY] = json.dumps(d)

        # the panel's footage info follows, but none of the panel's other settings are touched
        fr = scene.film_reel
        fr.source_type, fr.directory = src["type"], src["directory"]
        fr.files.clear()
        for nme in src["names"]:
            fr.files.add().name = nme
        fr.src_width, fr.src_height, fr.src_frames, fr.source_fps = sw, sh, sframes, sfps

        for area in context.screen.areas:
            area.tag_redraw()
        msg = "Replaced the footage on '%s' (%d frames, nothing else changed)%s" % (col.name, n, note)
        if held:
            msg += ". The new clip is shorter than the reel: its last frame fills the final %d frame(s)" % held
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def resize_reel_strip(strip, n_new):
    """Rebuild the strip's geometry in place for n_new frames, keeping the same mesh datablock,
    the reel's scale / cores / wind spacing and (through the object) its position. Returns the
    holder material. The caller re-fills the material slots."""
    uw = strip.film_reel_unwind
    mesh = strip.data
    verts, faces, mats, uvs = _decoded()
    pos, all_faces, lnorm, mat_idx, uv_all = strip_geometry(
        verts, faces, mats, uvs, uw.film_scale, n_new,
        'SPIRAL' if uw.built_spiral else 'STRAIGHT', uw.core_r, uw.wind)
    holder = mesh.materials[0] if len(mesh.materials) else get_holder_material()
    mesh.clear_geometry()
    mesh.from_pydata(pos.tolist(), [], all_faces)
    mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", uv_all)
    mesh.polygons.foreach_set("material_index", mat_idx.tolist())
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    try:
        mesh.normals_split_custom_set(lnorm.tolist())
    except Exception:
        pass
    mesh.update()
    uw.n_frames = n_new
    return holder


class FILMREEL_OT_replace_range(Operator):
    bl_idname = "film_reel.replace_frame_range"
    bl_label = "Replace Frame Range"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Re-adjust the frames of the selected reel (the one holding the active object, "
                      "otherwise the newest) to the From / To values set above. Only the reel's "
                      "frames change: its footage, position, cores, unwinding, flips, color space "
                      "and every other setting stay as they are")

    @classmethod
    def poll(cls, context):
        st = target_strip(context)
        return st is not None and st.film_reel_unwind.n_frames > 0

    def execute(self, context):
        scene = context.scene
        fr = scene.film_reel
        col = target_reel(context)
        strip = reel_strip(col)
        if strip is None:
            self.report({'ERROR'}, "No reel to update")
            return {'CANCELLED'}
        uw = strip.film_reel_unwind
        mesh = strip.data
        n_old = uw.n_frames
        if len(mesh.materials) != n_old + 1:
            self.report({'ERROR'}, "This reel's frame materials don't match its frame count")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # the reel's own settings and footage (older reels fall back to the panel)
        try:
            saved = json.loads(strip.get(_CFG_KEY) or "")
        except Exception:
            saved = json.loads(snapshot_settings(fr))
        skind = saved.get("source_type", 'NONE')
        sdir = saved.get("directory", "")
        snames = list(saved.get("_files", []))
        sw, sh = int(saved.get("src_width", 0)), int(saved.get("src_height", 0))
        sframes, sfps = int(saved.get("src_frames", 0)), float(saved.get("source_fps", 24.0))
        if skind not in ('SEQUENCE', 'VIDEO') or sframes <= 0 or sw <= 0 or sh <= 0:
            self.report({'ERROR'}, "This reel has no footage information saved")
            return {'CANCELLED'}
        preset = saved.get("fps_preset", '24')
        reel_fps = max(0.001, float(saved.get("custom_fps", 24.0))) if preset == 'CUSTOM' else float(preset)

        # the range asked for in the panel, limited to what the reel's footage can supply
        y = max(1, int(round(sframes / max(0.001, sfps) * reel_fps)))
        req = (fr.frame_start, fr.frame_end)
        first = min(max(1, fr.frame_start), y)
        last = min(max(first, fr.frame_end), y)
        n_new = last - first + 1
        ratio = sfps / reel_fps
        idx_map = [min(sframes - 1, int((first - 1 + j) * ratio + 1e-6)) for j in range(n_new)]

        if skind == 'SEQUENCE':
            missing = [snames[i] for i in sorted(set(idx_map))
                       if i >= len(snames) or not os.path.isfile(os.path.join(sdir, snames[i]))]
            if missing:
                self.report({'ERROR'}, "Missing frame file: %s" % missing[0])
                return {'CANCELLED'}
        elif not os.path.isfile(sdir):
            self.report({'ERROR'}, "Video file not found: %s" % sdir)
            return {'CANCELLED'}

        look = SimpleNamespace(aspect_preset=saved.get("aspect_preset", 'RECOMMENDED'),
                               custom_aspect=saved.get("custom_aspect", 16.0 / 9.0),
                               src_width=sw, src_height=sh)
        mapping = picture_mapping(sw, sh, picture_aspect(look), saved.get("image_fit", 'STRETCH'))
        mode = saved.get("max_image_size", 'RECOMMENDED')
        limit = (recommended_limit(skind, sw, sh, len(set(idx_map))) if mode == 'RECOMMENDED'
                 else int(mode))

        flips = reel_flip_state(col)             # remembered before the old pictures go
        cs = reel_colorspace(col)
        serial = reel_serial(col) or 1

        new_mats, new_images = [], {}
        try:
            new_mats, new_images = make_frame_materials(
                context, skind, sdir, snames, idx_map, limit,
                bool(saved.get("pack_images", False)), mapping, serial, first)
        except Exception as e:                   # nothing on the reel has been touched yet
            for m in new_mats:
                bpy.data.materials.remove(m)
            _purge_orphans()
            self.report({'ERROR'}, "Could not load the frames: %s" % e)
            return {'CANCELLED'}

        for axis, flipped in zip('HV', flips):
            if flipped:
                for m in new_mats:
                    flip_material(m, axis)
        if cs and cs != "Mixed":
            for img in new_images.values():
                try:
                    img.colorspace_settings.name = cs
                except Exception:
                    pass

        if n_new != n_old:                       # the reel gets longer / shorter: same object, new strip
            holder = resize_reel_strip(strip, n_new)
            mesh.materials.clear()
            mesh.materials.append(holder)
            for m in new_mats:
                mesh.materials.append(m)
        else:
            for k, m in enumerate(new_mats):
                mesh.materials[k + 1] = m
        apply_frame_order(strip)                 # keeps Reverse Film as it was
        _purge_orphans()                         # the old pictures
        for k, m in enumerate(new_mats):
            m.name = "M_FilmImage_R%d_%04d" % (serial, first + k)

        if uw.enabled:                           # an unwinding reel is re-bent for its new length
            _SIG.pop(strip.name, None)
            context.view_layer.update()
            update_unwind_reel(strip)

        # remember the new range on the reel (nothing else in its saved settings changes)
        try:
            d = json.loads(strip.get(_CFG_KEY) or "")
        except Exception:
            d = saved
        d.update(frame_start=first, frame_end=last)
        strip[_CFG_KEY] = json.dumps(d)

        _LOADING[0] = True                       # the panel shows the range that was really applied
        try:
            fr.frame_start, fr.frame_end = first, last
        finally:
            _LOADING[0] = False

        for area in context.screen.areas:
            area.tag_redraw()
        msg = "'%s' now holds frames %d-%d (%d frames, was %d)" % (col.name, first, last, n_new, n_old)
        if req != (first, last):
            msg += ". The footage only supplies %d frames at this reel's frame rate, so the range was limited" % y
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class FILMREEL_OT_recommended(Operator):
    bl_idname = "film_reel.recommended"
    bl_label = "Recommended"
    bl_description = ("Set the End Frame so the reel holds the recommended number of frames "
                      "(capped at %d for performance), counting from the Start Frame" % RECOMMENDED_MAX)

    def execute(self, context):
        fr = context.scene.film_reel
        first = effective_range(fr)[0]
        fr.frame_start = first
        fr.frame_end = min(yielded_frames(fr), first + RECOMMENDED_MAX - 1)
        return {'FINISHED'}


class FILMREEL_OT_all(Operator):
    bl_idname = "film_reel.all_frames"
    bl_label = "All Frames"
    bl_description = "Use the whole clip: every frame it yields at the chosen frame rate"

    def execute(self, context):
        fr = context.scene.film_reel
        fr.frame_start = 1
        fr.frame_end = yielded_frames(fr)
        return {'FINISHED'}


class FILMREEL_OT_render_size(Operator):
    bl_idname = "film_reel.set_render_size"
    bl_label = "Set Render Size"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Set the scene's render resolution to the chosen aspect ratio, using the "
                      "chosen resolution as the longest side")

    @classmethod
    def poll(cls, context):
        return context.scene.film_reel.src_frames > 0

    def execute(self, context):
        w, h = render_size_for(context.scene.film_reel)
        r = context.scene.render
        r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
        self.report({'INFO'}, "Render size set to %d x %d" % (w, h))
        return {'FINISHED'}


class FILMREEL_OT_build(Operator):
    bl_idname = "film_reel.build"
    bl_label = "Build Film Reel"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Create the film strip from the loaded footage"

    @classmethod
    def description(cls, context, properties):
        if context.scene.film_reel.build_mode == 'REPLACE':
            return ("Build the reel and replace an existing film reel made with this add-on, resetting "
                    "it completely to the current settings: the reel "
                    "that holds the active object, otherwise the most recently built one "
                    "(chosen with the Replace Reel option)")
        return "Build the reel as an additional film reel, keeping all existing ones (chosen with the Add New Reel option)"

    @classmethod
    def poll(cls, context):
        return context.scene.film_reel.src_frames > 0

    def execute(self, context):
        fr = context.scene.film_reel
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if fr.source_type == 'SEQUENCE':
            names = [f.name for f in fr.files]
            needed = sorted(set(source_index_map(fr)))
            missing = [names[i] for i in needed
                       if not os.path.isfile(os.path.join(fr.directory, names[i]))]
            if missing:
                self.report({'ERROR'}, "Missing frame file: %s" % missing[0])
                return {'CANCELLED'}
        elif not os.path.isfile(fr.directory):
            self.report({'ERROR'}, "Video file not found: %s" % fr.directory)
            return {'CANCELLED'}
        if fr.use_custom_ends and fr.layout_mode == 'SPIRAL':
            a, b = resolve_end_positions(fr)
            if (b - a).length < 1e-6:
                self.report({'ERROR'}, "Reel End A and End B are at the same position")
                return {'CANCELLED'}
        target = target_reel(context) if fr.build_mode == 'REPLACE' else None
        old_name = target.name if target is not None else ""
        try:
            n = build_reel(context, fr, target)
        except Exception as e:
            self.report({'ERROR'}, "Build failed: %s" % e)
            raise
        first, last = effective_range(fr)
        what = "Replaced '%s' with" % old_name if target is not None else "Built"
        self.report({'INFO'}, "%s a film reel of %d frames (%d-%d)" % (what, n, first, last))
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Color space editing (the "Color Space" dropdown on each frame's Image Texture node)
# ----------------------------------------------------------------------------
def reel_images(col):
    """Every distinct image used by the frame materials of a reel."""
    images, seen = [], set()
    for ob in col.all_objects:
        if ob.type != 'MESH':
            continue
        mats = [slot.material for slot in ob.material_slots] + list(ob.data.materials)
        for mat in mats:
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                img = getattr(node, "image", None)
                if img is not None and img.name not in seen:
                    seen.add(img.name)
                    images.append(img)
    return images


def reel_colorspace(col):
    """Color space shared by all frames of a reel, 'Mixed' if they differ, '' if none."""
    names = {img.colorspace_settings.name for img in reel_images(col)}
    if not names:
        return ""
    return names.pop() if len(names) == 1 else "Mixed"


_CS_ITEMS = []


def _colorspace_items(self, context):
    global _CS_ITEMS
    items = []
    try:
        reel = target_reel(context)
        imgs = reel_images(reel) if reel is not None else []
        if imgs:
            enum = imgs[0].colorspace_settings.bl_rna.properties["name"].enum_items
            items = [(e.identifier, e.name, e.description) for e in enum]
    except Exception:
        items = []
    if not items:
        items = [("sRGB", "sRGB", "")]
    _CS_ITEMS = items          # keep strings alive for Blender
    return _CS_ITEMS


class FILMREEL_OT_colorspace(Operator):
    bl_idname = "film_reel.set_colorspace"
    bl_label = "Set Frame Color Space"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Change the Color Space of every frame image of the reel (the one holding "
                      "the active object, otherwise the newest) in one go")

    colorspace: EnumProperty(name="Color Space", items=_colorspace_items,
                             description="Color space applied to all frame images")

    @classmethod
    def poll(cls, context):
        return target_reel(context) is not None

    def invoke(self, context, event):
        current = reel_colorspace(target_reel(context))
        if current and current != "Mixed":
            try:
                self.colorspace = current
            except Exception:
                pass
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        self.layout.prop(self, "colorspace")

    def execute(self, context):
        reel = target_reel(context)
        if reel is None:
            self.report({'ERROR'}, "No reel in this scene")
            return {'CANCELLED'}
        imgs = reel_images(reel)
        changed = 0
        for img in imgs:
            try:
                img.colorspace_settings.name = self.colorspace
                changed += 1
            except Exception as e:
                self.report({'WARNING'}, "%s: %s" % (img.name, e))
        for area in context.screen.areas:
            area.tag_redraw()
        self.report({'INFO'}, "Set color space '%s' on %d image(s) in '%s'" %
                    (self.colorspace, changed, reel.name))
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Flip the picture on every frame (done in each frame's Mapping node, so it is
# instant, works for video too, and never touches the image pixels)
# ----------------------------------------------------------------------------
def _frame_tex_node(mat):
    if mat is None or mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE'), None)


def _frame_mapping_node(mat, create=False):
    """The Mapping node feeding the picture texture (made on demand)."""
    tex = _frame_tex_node(mat)
    if tex is None:
        return None
    links = tex.inputs['Vector'].links
    if links and links[0].from_node.type == 'MAPPING':
        return links[0].from_node
    if not create:
        return None
    nt = mat.node_tree
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.location = (tex.location.x - 520, tex.location.y - 20)
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.location = (tex.location.x - 260, tex.location.y - 20)
    mp.vector_type = 'POINT'
    nt.links.new(uv.outputs['UV'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    return mp


def flip_material(mat, axis):
    """Mirror the picture of one frame material about the centre of the frame window.
    axis 'H' = left/right, 'V' = up/down. Applying it twice restores the original."""
    mp = _frame_mapping_node(mat, create=True)
    if mp is None:
        return False
    i = 0 if axis == 'H' else 1
    loc = list(mp.inputs['Location'].default_value)
    sc = list(mp.inputs['Scale'].default_value)
    loc[i] += sc[i]            # texture = (1 - uv) * s + o  ==  -s * uv + (s + o)
    sc[i] = -sc[i]
    mp.inputs['Location'].default_value = loc
    mp.inputs['Scale'].default_value = sc
    return True


def reel_frame_materials(col):
    mats, seen = [], set()
    strip = reel_strip(col)
    if strip is None:
        return mats
    for slot in strip.material_slots:
        m = slot.material
        if m is not None and m.name not in seen and _frame_tex_node(m) is not None:
            seen.add(m.name)
            mats.append(m)
    return mats


def reel_flip_state(col):
    """(flipped horizontally, flipped vertically) as seen on the reel's first frame."""
    for m in reel_frame_materials(col):
        mp = _frame_mapping_node(m)
        if mp is None:
            return False, False
        sc = mp.inputs['Scale'].default_value
        return sc[0] < 0.0, sc[1] < 0.0
    return False, False


class FILMREEL_OT_flip(Operator):
    bl_idname = "film_reel.flip_images"
    bl_label = "Flip Images"
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(items=[('H', "Horizontal", "Mirror left / right"),
                              ('V', "Vertical", "Mirror up / down")],
                       default='H', options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def description(cls, context, properties):
        return ("Flip the picture on every frame of the reel %s (click again to flip back)" %
                ("horizontally" if properties.axis == 'H' else "vertically"))

    @classmethod
    def poll(cls, context):
        return target_reel(context) is not None

    def execute(self, context):
        reel = target_reel(context)
        mats = reel_frame_materials(reel)
        if not mats:
            self.report({'WARNING'}, "No frame materials found in '%s'" % reel.name)
            return {'CANCELLED'}
        done = sum(1 for m in mats if flip_material(m, self.axis))
        for area in context.screen.areas:
            area.tag_redraw()
        self.report({'INFO'}, "Flipped %d frame(s) %s" %
                    (done, "horizontally" if self.axis == 'H' else "vertically"))
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Unwind operators
# ----------------------------------------------------------------------------
def target_strip(context):
    return reel_strip(target_reel(context))


def _unwind_poll(context):
    ob = target_strip(context)
    return ob is not None and ob.film_reel_unwind.built_spiral


class FILMREEL_OT_unwind_setup(Operator):
    bl_idname = "film_reel.unwind_setup"
    bl_label = "Set Up Unwind"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Let the wound reel unwind along a path and re-wind on a second reel. Creates "
                      "a take-up reel marker and a path curve you can move and reshape")

    @classmethod
    def poll(cls, context):
        return _unwind_poll(context)

    def execute(self, context):
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        create_unwind_helpers(context, ob)
        uw.enabled = True
        context.view_layer.update()
        _SIG.pop(ob.name, None)
        update_unwind_reel(ob)
        self.report({'INFO'}, "Unwind ready: drag the Unwind slider, or use Animate Over Timeline")
        return {'FINISHED'}


class FILMREEL_OT_unwind_refresh(Operator):
    bl_idname = "film_reel.unwind_refresh"
    bl_label = "Refresh"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Re-bend the film after you moved or reshaped the path, the take-up reel "
                      "or the reel itself (it also updates whenever the frame changes)")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        return ob is not None and ob.film_reel_unwind.enabled

    def execute(self, context):
        ob = target_strip(context)
        adopt_helpers(ob)
        context.view_layer.update()
        _SIG.pop(ob.name, None)
        update_unwind_reel(ob)
        return {'FINISHED'}


class FILMREEL_OT_unwind_step(Operator):
    bl_idname = "film_reel.unwind_step"
    bl_label = "Step Unwind"
    bl_options = {'REGISTER', 'UNDO'}

    direction: IntProperty(default=1, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def description(cls, context, properties):
        if properties.direction < 0:
            return "Move Unwind back by one Arrow Step (the step size is set below, in Unwind units)"
        return "Move Unwind forward by one Arrow Step (the step size is set below, in Unwind units)"

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        return ob is not None and ob.film_reel_unwind.enabled

    def execute(self, context):
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        sign = 1.0 if self.direction >= 0 else -1.0
        uw.progress = min(1.0, max(0.0, uw.progress + sign * uw.arrow_step))
        return {'FINISHED'}


class FILMREEL_OT_unwind_keyframe(Operator):
    bl_idname = "film_reel.unwind_keyframe"
    bl_label = "Set Unwind Keyframe"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Insert a keyframe for the Unwind value at the current frame of the timeline "
                      "(move the timeline first to choose where)")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        return ob is not None and ob.film_reel_unwind.enabled

    def execute(self, context):
        ob = target_strip(context)
        frame = context.scene.frame_current
        ob.keyframe_insert(data_path="film_reel_unwind.progress", frame=frame)
        self.report({'INFO'}, "Keyed Unwind = %.3f at frame %d" % (ob.film_reel_unwind.progress, frame))
        return {'FINISHED'}


class FILMREEL_OT_unwind_disable(Operator):
    bl_idname = "film_reel.unwind_disable"
    bl_label = "Back to Wound Reel"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Stop unwinding and restore the plain wound reel. The path and take-up "
                      "objects are kept")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        return ob is not None and ob.film_reel_unwind.enabled

    def execute(self, context):
        ob = target_strip(context)
        ob.film_reel_unwind.enabled = False
        restore_wound_reel(ob)
        return {'FINISHED'}


def reel_first_frame(strip):
    """Reel frame number of the first frame on the strip (the reel's From value)."""
    try:
        return max(1, int(json.loads(strip.get(_CFG_KEY) or "{}").get("frame_start", 1)))
    except Exception:
        return 1


def geometry_frame_for(strip, number):
    """Geometric frame index (0-based, along the strip) showing reel frame `number`.
    Reverse Film swaps which picture sits on which stretch of film."""
    uw = strip.film_reel_unwind
    i = min(max(number - reel_first_frame(strip), 0), uw.n_frames - 1)
    return (uw.n_frames - 1 - i) if uw.reverse_film else i


class FILMREEL_OT_unwind_align(Operator):
    bl_idname = "film_reel.unwind_align"
    bl_label = "Align Frame to Object"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Set Unwind so the centre of the chosen frame sits on the origin of the chosen "
                      "object. The frame can only travel along the film's route (round the supply "
                      "reel, along the path, round the take-up reel), so the object should lie on it: "
                      "otherwise the closest point is used and the miss is reported")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        if ob is None:
            return False
        uw = ob.film_reel_unwind
        return uw.enabled and uw.align_object is not None

    def execute(self, context):
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        tgt = uw.align_object
        if uw.path is None or uw.takeup is None:
            self.report({'ERROR'}, "Pick a path curve and a take-up object first")
            return {'CANCELLED'}
        if tgt is None or tgt == ob:
            self.report({'ERROR'}, "Pick an object (other than the reel) to align to")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.update()
        verts = _decoded()[0]
        Minv = np.array(ob.matrix_world.inverted())

        def to_local(P):                         # world points (m x 3) -> the reel's local space
            return P @ Minv[:3, :3].T + Minv[:3, 3]

        world = _curve_world_points(uw.path)
        if len(world) < 2:
            self.report({'ERROR'}, "The path curve has no points")
            return {'CANCELLED'}
        path = to_local(world)[:, 1:3]
        c_t = to_local(np.array([uw.takeup.matrix_world.translation[:]]))[0, 1:3]
        tl = to_local(np.array([tgt.matrix_world.translation[:]]))[0]
        k = geometry_frame_for(ob, uw.align_frame)
        p, d = align_progress(verts, uw.film_scale, uw.n_frames, uw.core_r, uw.wind,
                              uw.takeup_core_mm * uw.units_per_mm, path, c_t, k, tl[1:3])
        uw.progress = p
        context.view_layer.update()
        mm = uw.units_per_mm
        pitch_mm = _frame_pitch(verts, uw.film_scale) / mm
        miss, off = d / mm, abs(float(tl[0])) / mm
        first = reel_first_frame(ob)
        shown = first + ((uw.n_frames - 1 - k) if uw.reverse_film else k)
        msg = "Frame %d aligned to '%s' at Unwind = %.6f" % (shown, tgt.name, p)
        if miss > 0.1 * pitch_mm or off > 0.1 * pitch_mm:
            parts = []
            if miss > 0.1 * pitch_mm:
                parts.append("%.1f mm off the film's route" % miss)
            if off > 0.1 * pitch_mm:
                parts.append("%.1f mm to the side of the reel's plane" % off)
            self.report({'WARNING'}, msg + ". Closest approach only: the object is " + " and ".join(parts))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def _set_unwind_keys_linear(ob):
    """Make every key on the Unwind channel Linear (works on old and new Blender action APIs)."""
    ad = ob.animation_data
    act = ad.action if ad else None
    if act is None:
        return
    fcs = None
    try:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        fcs = cb.fcurves if cb is not None else None
    except Exception:
        fcs = None
    if fcs is None:
        fcs = getattr(act, "fcurves", [])
    for fc in fcs:
        if fc.data_path == "film_reel_unwind.progress":
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
            fc.update()


class FILMREEL_OT_unwind_key_all(Operator):
    bl_idname = "film_reel.unwind_key_all"
    bl_label = "Key Every Frame"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Put reel frame 1 at the Align Object on the scene's Start frame, frame 2 on the "
                      "next timeline frame, and so on: one Unwind key per timeline frame, all Linear. "
                      "The picture then holds still in front of the object for the whole animation. "
                      "Delete any older Unwind keys first")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        if ob is None:
            return False
        uw = ob.film_reel_unwind
        return uw.enabled and uw.align_object is not None

    def execute(self, context):
        sc = context.scene
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        if uw.path is None or uw.takeup is None:
            self.report({'ERROR'}, "Pick a path curve and a take-up object first")
            return {'CANCELLED'}
        first, n, start = reel_first_frame(ob), uw.n_frames, sc.frame_start
        cur, old_align = sc.frame_current, uw.align_frame
        try:
            for i in range(n):
                sc.frame_set(start + i)
                uw.align_frame = first + i
                if bpy.ops.film_reel.unwind_align() != {'FINISHED'}:
                    self.report({'ERROR'}, "Align failed at reel frame %d" % (first + i))
                    return {'CANCELLED'}
                ob.keyframe_insert(data_path="film_reel_unwind.progress", frame=start + i)
        finally:
            uw.align_frame = old_align
            sc.frame_set(cur)
        _set_unwind_keys_linear(ob)
        self.report({'INFO'}, "Keyed %d reel frames on timeline frames %d-%d (Linear)"
                    % (n, start, start + n - 1))
        return {'FINISHED'}


class FILMREEL_OT_unwind_pick_frame(Operator):
    bl_idname = "film_reel.unwind_pick_frame"
    bl_label = "Pick Frame from Selected Face"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("In Edit Mode, select a face of the frame you want and click this to fill in "
                      "the Frame number from it")

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return (context.mode == 'EDIT_MESH' and ob is not None and ob.get(MARK)
                and ob.film_reel_unwind.n_frames > 0)

    def execute(self, context):
        import bmesh
        ob = context.edit_object
        uw = ob.film_reel_unwind
        bm = bmesh.from_edit_mesh(ob.data)
        bm.faces.index_update()
        sel = [f.index for f in bm.faces if f.select]
        if not sel:
            self.report({'ERROR'}, "Select a face of the frame you want")
            return {'CANCELLED'}
        nf = len(bm.faces) // uw.n_frames
        if nf <= 0 or len(bm.faces) % uw.n_frames:
            self.report({'ERROR'}, "The strip's faces no longer match its frame count")
            return {'CANCELLED'}
        ks = sorted({i // nf for i in sel})
        k = ks[0]
        uw.align_frame = reel_first_frame(ob) + ((uw.n_frames - 1 - k) if uw.reverse_film else k)
        if len(ks) > 1:
            self.report({'WARNING'}, "Selection spans %d frames: using frame %d" % (len(ks), uw.align_frame))
        else:
            self.report({'INFO'}, "Frame %d picked" % uw.align_frame)
        return {'FINISHED'}


class FILMREEL_OT_path_subdivide(Operator):
    bl_idname = "film_reel.path_subdivide"
    bl_label = "Subdivide Path"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Re-divide the film path into the chosen number of segments, evenly by length, "
                      "keeping its shape. Each segment end becomes a control point you can move")

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        return ob is not None and ob.film_reel_unwind.path is not None

    def execute(self, context):
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        pob = uw.path
        pts = _curve_local_points(pob)
        if len(pts) < 2:
            self.report({'ERROR'}, "The path curve has no points")
            return {'CANCELLED'}
        n = int(uw.path_segments)
        new = resample_polyline(pts, n)
        set_path_points(pob, new, smooth=n > 1)
        context.view_layer.update()
        _SIG.pop(ob.name, None)
        update_unwind_reel(ob)
        self.report({'INFO'}, "Path divided into %d segment%s (%d points)" % (n, "" if n == 1 else "s", n + 1))
        return {'FINISHED'}


class FILMREEL_OT_path_fix(Operator):
    bl_idname = "film_reel.path_fix"
    bl_label = "Move Path Outside Reel"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Move the path curve's points that sit inside a wound reel (at its full size) "
                      "to just outside it, straight out from the reel's centre. Only the points "
                      "inside a reel move; no other setting is changed")
    MARGIN = 1.02          # how far outside the full reel the points are placed (1.0 = on its surface)

    @classmethod
    def poll(cls, context):
        ob = target_strip(context)
        if ob is None:
            return False
        uw = ob.film_reel_unwind
        return uw.enabled and uw.path is not None and uw.takeup is not None

    def execute(self, context):
        ob = target_strip(context)
        uw = ob.film_reel_unwind
        pob = uw.path
        if pob.type != 'CURVE' or not pob.data.splines:
            self.report({'ERROR'}, "The path curve has no points")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.update()

        pitch = _frame_pitch(_decoded()[0], uw.film_scale)
        L = uw.n_frames * pitch
        r_s = _full_radius(uw.core_r, uw.wind, L, pitch) * self.MARGIN
        r_t = _full_radius(uw.takeup_core_mm * uw.units_per_mm, uw.wind, L, pitch) * self.MARGIN
        M = np.array(ob.matrix_world)
        Minv = np.linalg.inv(M)
        Pm = np.array(pob.matrix_world)
        P3inv = np.linalg.inv(Pm[:3, :3])
        c_t = (Minv @ np.append(np.array(uw.takeup.matrix_world.translation[:]), 1.0))[1:3]
        circles = [(np.zeros(2), r_s), (c_t, r_t)]

        def shift(co3):
            """Offset (in the curve's own space) that takes a point out of both reels, or None.
            Only the position in the reel's plane changes; the offset along the reel axis stays."""
            w = Pm[:3, :3] @ co3 + Pm[:3, 3]
            q = Minv[:3, :3] @ w + Minv[:3, 3]
            yz = q[1:3].copy()
            moved = False
            for _ in range(2):                       # twice, in case the two reels overlap
                for k, (c, r) in enumerate(circles):
                    v = yz - c
                    d = float(np.hypot(v[0], v[1]))
                    if d >= r:
                        continue
                    if d > 1e-12:
                        u = v / d
                    else:                            # exactly on the centre: leave away from the other reel
                        v = c - circles[1 - k][0]
                        d = float(np.hypot(v[0], v[1]))
                        u = v / d if d > 1e-12 else np.array([0.0, 1.0])
                    yz = c + u * r
                    moved = True
            if not moved:
                return None
            q[1:3] = yz
            w2 = M[:3, :3] @ q + M[:3, 3]
            return P3inv @ (w2 - w)

        n_moved = 0
        for sp in pob.data.splines:
            if sp.type == 'BEZIER':
                for bp in sp.bezier_points:
                    d = shift(np.array(bp.co[:]))
                    if d is None:
                        continue
                    bp.co = tuple(float(x) for x in np.array(bp.co[:]) + d)
                    bp.handle_left = tuple(float(x) for x in np.array(bp.handle_left[:]) + d)
                    bp.handle_right = tuple(float(x) for x in np.array(bp.handle_right[:]) + d)
                    n_moved += 1
            else:
                for pt in sp.points:
                    d = shift(np.array(pt.co[:3]))
                    if d is None:
                        continue
                    pt.co = (pt.co[0] + float(d[0]), pt.co[1] + float(d[1]), pt.co[2] + float(d[2]), pt.co[3])
                    n_moved += 1

        pob.data.update_tag()
        context.view_layer.update()
        _SIG.pop(ob.name, None)
        update_unwind_reel(ob)
        for area in context.screen.areas:
            area.tag_redraw()
        if n_moved == 0:
            self.report({'INFO'}, "No path points were inside a reel")
        elif uw.status:
            self.report({'WARNING'}, "Moved %d path point(s), but: %s" % (n_moved, uw.status))
        else:
            self.report({'INFO'}, "Moved %d path point(s) just outside the reel" % n_moved)
        return {'FINISHED'}


class FILMREEL_OT_set_ends(Operator):
    bl_idname = "film_reel.set_end_positions"
    bl_label = "Set Reel End Positions"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("Turn on custom reel end positions. With exactly 2 objects selected, their "
                      "origins become the ends (the active object is End A, the supply reel; the other "
                      "is End B, the take-up reel). Otherwise the typed coordinates are used")

    def execute(self, context):
        fr = context.scene.film_reel
        sel = list(context.selected_objects)
        if len(sel) == 2:
            act = context.view_layer.objects.active
            a = act if act in sel else sel[0]
            b = sel[1] if sel[0] == a else sel[0]
            fr.end_a = a.matrix_world.translation[:]      # (clears the picked objects...)
            fr.end_b = b.matrix_world.translation[:]
            fr.end_obj_a, fr.end_obj_b = a, b             # ...so set them afterwards
            msg = "End A = '%s', End B = '%s'" % (a.name, b.name)
        elif not sel:
            msg = "Using the typed coordinates"
        else:
            self.report({'ERROR'}, "Select exactly 2 objects (or none, to use the typed coordinates)")
            return {'CANCELLED'}
        fr.use_custom_ends = True
        self.report({'INFO'}, "Reel ends set: " + msg + ". They are applied when you click Build")
        return {'FINISHED'}


class FILMREEL_OT_reset_ends(Operator):
    bl_idname = "film_reel.reset_end_positions"
    bl_label = "Reset"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Reset End A and End B to their default positions and clear any picked objects"

    def execute(self, context):
        fr = context.scene.film_reel
        fr.end_a = fr.bl_rna.properties["end_a"].default_array
        fr.end_b = fr.bl_rna.properties["end_b"].default_array
        fr.end_obj_a = None                       # typing coordinates clears these too, but be sure
        fr.end_obj_b = None
        return {'FINISHED'}


class FILMREEL_OT_swap_ends(Operator):
    bl_idname = "film_reel.swap_end_positions"
    bl_label = "Swap"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Swap End A (supply reel) and End B (take-up reel)"

    def execute(self, context):
        fr = context.scene.film_reel
        oa, ob_ = fr.end_obj_a, fr.end_obj_b
        a, b = tuple(fr.end_a), tuple(fr.end_b)
        fr.end_a, fr.end_b = b, a
        fr.end_obj_a, fr.end_obj_b = ob_, oa
        return {'FINISHED'}


def draw_core_editors(layout, fr, uw):
    col = layout.column(align=True)
    col.use_property_split = True
    col.label(text="Reel Cores", icon='MESH_CIRCLE')
    if uw is not None and uw.built_spiral:          # live: re-bends the reel as you change it
        col.prop(uw, "supply_core_mm", text="End A (Supply)")
        col.prop(uw, "takeup_core_mm", text="End B (Take-up)")
    else:                                           # no wound reel yet: values for the next build
        col.prop(fr, "core_radius_mm", text="End A (Supply)")
        col.prop(fr, "takeup_core_mm", text="End B (Take-up)")
    layout.separator(factor=0.5)


def draw_unwind_ui(layout, context):
    box = layout.box()
    box.label(text="Unwind Along Path", icon='CURVE_PATH')
    ob = target_strip(context)
    draw_core_editors(box, context.scene.film_reel, ob.film_reel_unwind if ob is not None else None)
    if ob is None:
        box.label(text="Build a reel first", icon='INFO')
        return
    uw = ob.film_reel_unwind
    if not uw.built_spiral:
        box.label(text="Rebuild this reel as a Wound Reel", icon='INFO')
        return
    if not uw.enabled:
        box.prop(uw, "path_segments")
        box.operator("film_reel.unwind_setup", icon='ADD')
        box.label(text="Reel: %s" % ob.name, icon='INFO')
        return
    row = box.row(align=False)
    sub = row.row(align=True)
    sub.operator("film_reel.unwind_step", text="", icon='TRIA_LEFT').direction = -1
    sub.prop(uw, "progress", text="Unwind", slider=False)
    sub.operator("film_reel.unwind_step", text="", icon='TRIA_RIGHT').direction = 1
    row.operator("film_reel.unwind_keyframe", text="", icon='KEY_HLT')
    col = box.column(align=True)
    col.use_property_split = True
    col.prop(uw, "arrow_step")
    box.separator(factor=0.5)
    col = box.column(align=True)
    col.use_property_split = True
    col.prop(uw, "path")
    col.prop(uw, "takeup")
    col.prop(uw, "reverse_film")
    row = box.row(align=True)
    row.prop(uw, "path_segments", text="Path Subdivisions")
    row.operator("film_reel.path_subdivide", text="Subdivide", icon='MOD_SUBSURF')
    if uw.path is None or uw.takeup is None:
        box.label(text="Pick a path curve and a take-up object", icon='ERROR')
    elif uw.status:
        box.label(text=uw.status, icon='ERROR')
        box.operator("film_reel.path_fix", icon='SNAP_ON')
    sub = box.box()
    sub.label(text="Align Frame to Object", icon='SNAP_ON')
    first = reel_first_frame(ob)
    row = sub.row(align=True)
    row.prop(uw, "align_frame", text="Frame (%d-%d)" % (first, first + uw.n_frames - 1))
    row.operator("film_reel.unwind_pick_frame", text="", icon='RESTRICT_SELECT_OFF')
    sub.prop(uw, "align_object", text="Object")
    sub.operator("film_reel.unwind_align", icon='SNAP_ON')
    sub.operator("film_reel.unwind_key_all", icon='KEY_HLT')
    row = box.row(align=True)
    row.operator("film_reel.unwind_refresh", icon='FILE_REFRESH')
    row.operator("film_reel.unwind_disable", icon='X')
    box.label(text="Reel: %s" % ob.name, icon='INFO')


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
class FILMREEL_PT_main(Panel):
    bl_label = "Film Reel"
    bl_idname = "FILMREEL_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Film Reel"

    def draw_header(self, context):
        self.layout.label(icon='SEQUENCE')

    def draw(self, context):
        layout = self.layout
        fr = context.scene.film_reel
        layout.operator("film_reel.load_source", icon='FILE_MOVIE')
        if target_reel(context) is not None:
            layout.operator("film_reel.replace_source", icon='FILE_REFRESH')
        if fr.src_frames <= 0:
            box = layout.box()
            box.label(text="Load a video or image sequence to begin", icon='INFO')
            return

        kind = "Video" if fr.source_type == 'VIDEO' else "Image sequence"
        box = layout.box()
        col = box.column(align=True)
        col.label(text="%s: %dx%d px, %d frames" % (kind, fr.src_width, fr.src_height, fr.src_frames),
                  icon='FILE_MOVIE')
        col.label(text="%.2f s at %.3f fps" % (fr.src_frames / max(0.001, fr.source_fps), fr.source_fps))
        col.prop(fr, "source_fps")

        layout.separator()
        layout.label(text="Frame Rate (fps)", icon='TIME')
        grid = layout.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        grid.prop(fr, "fps_preset", expand=True)
        if fr.fps_preset == 'CUSTOM':
            layout.prop(fr, "custom_fps")

        layout.separator()
        layout.label(text="Film Size", icon='SEQUENCE')
        grid = layout.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        grid.prop(fr, "film_size", expand=True)
        wmm = FILM_WIDTH[fr.film_size]
        layout.label(text="%s: %g mm wide = %.4g m in Blender" % (FILM_LABEL[fr.film_size], wmm,
                                                                  film_width_m(fr)), icon='INFO')
        if fr.film_size != '35MM':
            layout.label(text="(35mm 4-perf model, scaled to width)")

        layout.separator()
        layout.label(text="Picture Format", icon='IMAGE_DATA')
        layout.prop(fr, "aspect_preset", text="Aspect Ratio")
        if fr.aspect_preset == 'CUSTOM':
            layout.prop(fr, "custom_aspect", text="Width / Height")
        asp, src_asp = picture_aspect(fr), source_aspect(fr)
        if fr.aspect_preset == 'RECOMMENDED':
            layout.label(text="Matches source: %s" % aspect_text(asp), icon='CHECKMARK')
        else:
            layout.label(text="Picture %s - source is %s" % (aspect_text(asp), aspect_text(src_asp)),
                         icon='INFO')
            layout.prop(fr, "image_fit", text="Fit")
        col = layout.column()
        col.enabled = fr.source_type != 'VIDEO'
        col.prop(fr, "max_image_size", text="Resolution")
        tw, th = target_dims(fr)
        if fr.source_type == 'VIDEO':
            layout.label(text="Video decodes at native %dx%d px" % (tw, th), icon='INFO')
        else:
            rec = recommended_resolution(fr)
            if fr.max_image_size == 'RECOMMENDED':
                layout.label(text="Recommended: %s" % ("Original" if rec == 0 else "%d px" % rec),
                             icon='CHECKMARK')
            layout.label(text="Frame textures: %dx%d px (source %dx%d)" %
                         (tw, th, fr.src_width, fr.src_height))
        rw, rh = render_size_for(fr)
        layout.operator("film_reel.set_render_size",
                        text="Set Render Size (%dx%d)" % (rw, rh), icon='OUTPUT')

        layout.separator()
        layout.label(text="Frame Range", icon='SEQ_STRIP_META')
        row = layout.row(align=True)
        row.prop(fr, "frame_start", text="From")
        row.prop(fr, "frame_end", text="To")
        y = yielded_frames(fr)
        first, last = effective_range(fr)
        if (fr.frame_start, fr.frame_end) != (first, last):
            layout.label(text="Clip has only %d frames: using %d-%d" % (y, first, last), icon='ERROR')
        row = layout.row(align=True)
        row.operator("film_reel.recommended")
        row.operator("film_reel.all_frames")
        if target_reel(context) is not None:
            layout.operator("film_reel.replace_frame_range", icon='FILE_REFRESH')
        layout.label(text="%s yields %d frames (recommended: %d)" %
                     ("Video" if fr.source_type == 'VIDEO' else "Sequence", y, recommended_frames(fr)))
        gb = memory_gb(fr)
        layout.label(text="~%.2f GB image memory" % gb, icon='CHECKMARK' if gb < 2.0 else 'ERROR')
        n = effective_frames(fr)
        pitch = pitch_mm_for(fr.film_size)
        total = n * pitch
        layout.label(text="Reel: %d frames, %.0f mm (%.1f ft)" % (n, total, total / 304.8))

        layout.separator()
        box = layout.box()
        box.prop(fr, "use_custom_ends", icon='PIVOT_CURSOR')
        if fr.use_custom_ends:
            if fr.layout_mode != 'SPIRAL':
                box.label(text="Applies to the Wound Reel layout", icon='INFO')
            col = box.column(align=True)
            col.prop(fr, "end_a", text="A")
            col.prop(fr, "end_b", text="B")
            row = box.row(align=True)
            row.prop(fr, "end_obj_a", text="")
            row.prop(fr, "end_obj_b", text="")
            if fr.end_obj_a or fr.end_obj_b:
                box.label(text="Picked objects are read when you click Build", icon='INFO')
            row = box.row(align=True)
            row.operator("film_reel.set_end_positions", icon='EYEDROPPER')
            row.operator("film_reel.swap_end_positions", icon='ARROW_LEFTRIGHT')
            box.operator("film_reel.reset_end_positions", icon='LOOP_BACK',
                         text="Reset End Positions to Default")

        target = target_reel(context)
        row = layout.row(align=True)
        row.scale_y = 1.4
        sub = row.row(align=True)
        sub.enabled = target is not None
        sub.prop_enum(fr, "build_mode", 'REPLACE', text="Replace Reel", icon='FILE_REFRESH')
        row.prop_enum(fr, "build_mode", 'ADD', text="Add New Reel", icon='ADD')
        if target is None:
            layout.label(text="No reel in this scene yet: Build adds a new reel", icon='INFO')
        elif fr.build_mode == 'REPLACE':
            layout.label(text="Build Film Reel will reset and replace: %s" % target.name, icon='INFO')
        else:
            layout.label(text="Build Film Reel will add a new reel", icon='INFO')
        layout.operator("film_reel.build", icon='SEQUENCE')

        if target is not None:
            layout.separator()
            layout.label(text="Frame Color Space", icon='COLOR')
            cs = reel_colorspace(target)
            layout.label(text="Current: %s" % (cs or "not found"), icon='INFO')
            layout.operator("film_reel.set_colorspace", text="Change Color Space...",
                            icon='IMAGE_DATA')

            layout.separator()
            layout.label(text="Flip Images", icon='MOD_MIRROR')
            fh, fv = reel_flip_state(target)
            row = layout.row(align=True)
            row.operator("film_reel.flip_images", text="Horizontal", icon='ARROW_LEFTRIGHT',
                         depress=fh).axis = 'H'
            row.operator("film_reel.flip_images", text="Vertical", icon='EMPTY_SINGLE_ARROW',
                         depress=fv).axis = 'V'


class FILMREEL_PT_options(Panel):
    bl_label = "Reel Options"
    bl_idname = "FILMREEL_PT_options"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Film Reel"
    bl_parent_id = "FILMREEL_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        fr = context.scene.film_reel
        layout.prop(fr, "layout_mode")
        if fr.layout_mode == 'STRAIGHT':
            layout.prop(fr, "stand_upright")
        else:
            _st = target_strip(context)
            if _st is not None and _st.film_reel_unwind.built_spiral:
                layout.prop(_st.film_reel_unwind, "wind_mm", text="Wind Spacing")
            else:
                layout.prop(fr, "wind_spacing_mm")
            layout.separator()
            draw_unwind_ui(layout, context)
        layout.separator()
        sub = layout.column()
        sub.enabled = fr.source_type != 'VIDEO'
        sub.prop(fr, "pack_images")
        layout.separator()
        layout.prop(fr, "scale_multiplier")
        upm = units_per_mm(context.scene, fr)
        layout.label(text="1 mm of film = %.6g Blender units" % upm, icon='INFO')
        if fr.scale_multiplier == 1.0:
            layout.label(text="True scale (follows Scene Unit Scale)", icon='CHECKMARK')


classes = (FilmReelFile, FilmReelSettings, FilmReelUnwind, FILMREEL_OT_load, FILMREEL_OT_replace_source, FILMREEL_OT_replace_range, FILMREEL_OT_recommended,
           FILMREEL_OT_all, FILMREEL_OT_render_size, FILMREEL_OT_build, FILMREEL_OT_colorspace, FILMREEL_OT_flip,
           FILMREEL_OT_unwind_setup, FILMREEL_OT_unwind_refresh, FILMREEL_OT_unwind_step,
           FILMREEL_OT_unwind_keyframe, FILMREEL_OT_path_subdivide, FILMREEL_OT_path_fix,
           FILMREEL_OT_set_ends, FILMREEL_OT_swap_ends, FILMREEL_OT_reset_ends,
           FILMREEL_OT_unwind_align, FILMREEL_OT_unwind_key_all, FILMREEL_OT_unwind_pick_frame,
           FILMREEL_OT_unwind_disable, FILMREEL_PT_main, FILMREEL_PT_options)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.film_reel = PointerProperty(type=FilmReelSettings)
    bpy.types.Object.film_reel_unwind = PointerProperty(type=FilmReelUnwind)
    if _unwind_frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_unwind_frame_handler)
    if _unwind_depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_unwind_depsgraph_handler)
    if _follow_depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_follow_depsgraph_handler)
    if _follow_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_follow_load_post)
    _FOLLOW["reel"] = None
    _subscribe_active()


def unregister():
    if _follow_depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_follow_depsgraph_handler)
    if _follow_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_follow_load_post)
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    except Exception:
        pass
    if _unwind_depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_unwind_depsgraph_handler)
    if _unwind_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_unwind_frame_handler)
    del bpy.types.Object.film_reel_unwind
    del bpy.types.Scene.film_reel
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()

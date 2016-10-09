import numpy as np
import cv2
import random
import os
import shutil
import math
import argparse
import sys




def generate_star(size, R, G, B):

   #Nahodne vygenerujeme primku
    ax = random.randint(int(size/2 - (2*size/2)/math.sqrt(1/math.tan(math.pi/10)**2 + 4)),size/2 -
                         int(args['min_size'] * math.sin(math.pi/10)))
    bx = size - ax

    #Dopocitame vysku hvezdy
    h = int((bx - ax)/(2*math.tan(math.pi/10)))

    #Podle vysky zvolime pozici primky, tak aby hvezda lezel ve stredu
    aby = size - (size - h)/2

    #Dopocitame vrchol hvezdy
    cx = ax + (bx - ax)/2
    cy = aby - h

    zx = cx
    zy = int(aby - math.tan(math.pi/5)*((bx - ax)/2))

    l1 = (bx - ax)/(2*(math.cos(math.pi/5)))
    s = (bx - ax)/(2*(math.sin(math.pi/10)))
    l2 = s - 2*l1

    h1 = math.sin(math.pi/5)*(l1 + l2)
    w = math.sqrt((l1 + l2)**2 - h1**2)


    #Krajni body hvezdy
    ix = int(cx - w)
    iy = int(zy - h1)

    jx = int(cx + w)
    jy = iy



    tri = np.array([[ix,iy], [jx,jy], [zx,zy]])
    arrow = np.array([[ax,aby], [zx,zy], [bx,aby], [cx,cy]])

    img = np.zeros((size,size,3), np.uint8)

    img = cv2.fillPoly(img, [tri], (R,G,B))
    img = cv2.fillPoly(img, [arrow], (R,G,B))

    try:
        shiftx = random.randint(-(int(size/2 - math.sqrt(((bx - ax)/2)**2 + (h/2)**2))),int(size/2 - math.sqrt(((bx - ax)/2)**2 + (h/2)**2)))
        shifty = random.randint(-(int(size/2 - math.sqrt(((bx - ax)/2)**2 + (h/2)**2))),int(size/2 - math.sqrt(((bx - ax)/2)**2 + (h/2)**2)))
    except:
        shiftx = 0
        shifty = 0


    return (img, shiftx, shifty, math.sqrt(((bx - ax)/2)**2 + (h/2)**2)*2)


def generate_round(size, R, G, B):

    #Stred ve stredu
    x = size/2
    y = size/2

    #Nahodne vygenerujeme polomer
    r = random.randint(args['min_size']/2, size/2)

    img = np.zeros((size,size,3), np.uint8)
    #img[:] = (0, 0, 255)

    img = cv2.circle(img, (x,y), r, (R,G,B), cv2.FILLED)
    img = cv2.circle(img, (x,y), int(r-r/3), (0,0,0), cv2.FILLED)

    shiftx = random.randint(-(size/2 - r),size/2 - r)
    shifty = random.randint(-(size/2 - r),size/2 - r)

    return (img, shiftx, shifty, 2*r)

def generate_cross(size, R, G, B):
    #Nahodne generujeme prvni bod v II. kvadrantu
    x11 = random.randint(size/2 - int((2*size)/math.sqrt(17)), size/2 - args['min_size']/2)
    y11 = size/2 - (size/2 - x11)/4

    x12 = size - x11
    y12 = y11 + (size/2 - x11)/2

    x21 = y11
    y21 = x11

    x22 = y12
    y22 = x12

    #Vygenerujeme nahodny ctverec
    img = np.zeros((size,size,3), np.uint8)
    #img[:] = (0, 0, 255)
    img = cv2.rectangle(img,(x11,y11),(x12,y12),(R,G,B), cv2.FILLED)
    img = cv2.rectangle(img,(x21,y21),(x22,y22),(R,G,B), cv2.FILLED)



    shiftx = random.randint(-(size/2 - int(math.sqrt((size/2 - x11)**2 + ((size/2 - x11)/4)**2))),size/2 - int(math.sqrt((size/2 - x11)**2 + ((size/2 - x11)/4)**2)))
    shifty = random.randint(-(size/2 - int(math.sqrt((size/2 - x11)**2 + ((size/2 - x11)/4)**2))),size/2 - int(math.sqrt((size/2 - x11)**2 + ((size/2 - x11)/4)**2)))

    return (img, shiftx, shifty, (math.sqrt((size/2 - x11)**2 + ((size/2 - x11)/4)**2))*2)



def generate_square(size, R, G, B):

    #Nahodne generujeme prvni bod v II. kvadrantu
    if args['random_size']:
        xy = random.randint(size/2 - int(math.sqrt(((size/2)**2)/2)), size/2 - int(math.sqrt((args['min_size']**2)/2)/2))
    else:
        xy = size/2 - int(math.sqrt((args['min_size']**2)/2)/2)
    #Dopocitame delku ctverce
    s = size - 2*xy;

    #Vygenerujeme nahodny ctverec
    img = np.zeros((size,size,3), np.uint8)
    #img[:] = (0, 0, 255)
    img = cv2.rectangle(img,(xy,xy),(xy+s,xy+s),(R,G,B), cv2.FILLED)

    shiftx = random.randint(-(size/2 - int(math.sqrt(((s/2)**2)*2))),size/2 - int(math.sqrt(((s/2)**2)*2)))
    shifty = random.randint(-(size/2 - int(math.sqrt(((s/2)**2)*2))),size/2 - int(math.sqrt(((s/2)**2)*2)))

    return (img, shiftx, shifty, math.sqrt(((s)**2)*2))

def generate_circle(size, R, G, B):

    #Stred ve stredu
    x = size/2
    y = size/2

    #Nahodne vygenerujeme polomer
    if args['random_size']:
        r = random.randint(args['min_size']/2, size/2)
    else:
        r = args['min_size']/2

    img = np.zeros((size,size,3), np.uint8)
    img = cv2.circle(img, (x,y), r, (R,G,B), cv2.FILLED)

    shiftx = random.randint(-(size/2 - r),size/2 - r)
    shifty = random.randint(-(size/2 - r),size/2 - r)

    return (img, shiftx, shifty, 2*r)

def generate_triangle(size, R, G, B):

    #Nahodne vygenerujeme primku
    if args['random_size']:
        ax = random.randint(int(size/2*(1 - math.cos(math.pi/6))), size/2 - args['min_size']/2)
    else:
        ax =  size/2 - args['min_size']/2
    bx = size - ax

    #Dopocitame vysku rovnostraneho trojuhelniku
    h = int(math.sqrt((bx - ax)**2 - ((bx - ax)/2)**2))

    #Podle vysky zvolime pozici primky, tak aby trojuhelnik lezel ve stredu
    aby = size/2 + h/3

    #Dopocitame posledni bod trojuhelniku
    cx = ax + (bx - ax)/2
    cy = aby - h

    triangle = np.array([[ax,aby],[bx,aby], [cx,cy]])

    img = np.zeros((size,size,3), np.uint8)

    img = cv2.fillPoly(img, [triangle], (R,G,B))


    try:
        shiftx = random.randint(-(size/2 - 2*h/3),size/2 - 2*h/3)
        shifty = random.randint(-(size/2 - 2*h/3),size/2 - 2*h/3)
    except:
        shiftx = 0
        shifty = 0

    return (img, shiftx, shifty, 4*h/3)

def transform(img, shiftx, shifty):

    rx = random.randint(-60, 60)
    ry = random.randint(-60, 60)
    rz = random.randint(0, 360)

    radang = math.pi*rz/180

    xrot = math.sin(radang)
    yrot = math.cos(radang)

    fov = random.randint(1, 60)

    if (not args['position']):
        img_prop = (warpPerspective(rx, ry, rz, fov, img), xrot, yrot, rx, ry, fov)
    if (not args['position'] and not args['transformation']):
        img_prop = (img, xrot, yrot, rx, ry, fov)

    return (img_prop)



def warpPerspective(rx, ry, rz, fov, img, positions=None, shift=(0,0)):

    s = max(img.shape[0:2])
    rotVec = np.asarray((rx*np.pi/180,ry*np.pi/180, rz*np.pi/180))
    rotMat, j = cv2.Rodrigues(rotVec)
    rotMat[0, 2] = 0
    rotMat[1, 2] = 0
    rotMat[2, 2] = 1

    f = 0.3
    trnMat1 = np.asarray(
        (1, 0, -img.shape[1]/2,
         0, 1, -img.shape[0]/2,
         0, 0, 1)).reshape(3, 3)

    T1 = np.dot(rotMat, trnMat1)
    distance = (s/2)/math.tan(fov*np.pi/180)
    T1[2, 2] += distance


    cameraT = np.asarray(
        (distance, 0, img.shape[1]/2 + shift[1],
         0, distance, img.shape[0]/2 + shift[0],
         0, 0, 1)).reshape(3,3)

    T2 = np.dot(cameraT, T1)

    newImage = cv2.warpPerspective(img, T2, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_NEAREST)
    if positions is None:
        return newImage
    else:
        return newImage, np.squeeze( cv2.perspectiveTransform(positions[None, :, :], T2), axis=0)

    return rotatedImg


def create_set(obj_type):

    if obj_type:
        dir = args['test_out']
        num = args['test_size']
    else:
        dir = args['train_out']
        num = args['train_size']

    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)

    if os.path.isfile(os.path.join(os.getcwd(), dir + '.txt')):
        os.remove(os.path.join(os.getcwd(), dir + '.txt'))
    f=open(os.path.join(os.getcwd(), dir + '.txt'), 'w+')

    for x in range (0, num):

        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)

        new_object = random.randint(0, 2)

        tri = 0.0
        sqr = 0.0
        cir = 0.0
        cro = 0.0
        rnd = 0.0
        sta = 0.0

        if new_object == 0:
            img_prop = generate_circle(args['super_size'], R, G, B)
            cir = 1.0
        elif new_object == 1:
            img_prop = generate_square(args['super_size'], R, G, B)
            sqr = 1.0
        elif new_object == 2:
            img_prop = generate_triangle(args['super_size'], R, G, B)
            tri = 1.0

        trans = transform(img_prop[0], img_prop[1], img_prop[2])

        img = trans[0]

        img = cv2.resize(img, (args['size'], args['size']), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dir, dir + '_%d.png' % x), img)


        f.write('%s_%d.png %f %f %f' % (dir, x, cir, sqr, tri))

        if args['colour']:
            f.write(' %f %f %f' % (float(R)/255.0, float(G)/255.0, float(B)/255.0))

        if args['position']:
            f.write(' %f %f' % (img_prop[1]/float(args['super_size']), img_prop[2]/float(args['super_size'])))

        if args['zoom']:
            f.write(' %f' % (img_prop[3]/float(args['super_size'])))

        if args['rotation']:
            f.write(' %f %f' % (trans[1], trans[2]))

        if args['transformation']:
            f.write(' %f %f %f' % (trans[3]/60.0, trans[4]/60.0, trans[5]/60.0))

        f.write('\n')


    f.close()




args_parser = argparse.ArgumentParser(description='Generovani jednoduchych objektu')
args_parser.add_argument('-s','--size', type = int, help = 'Sirka a vyska vysledneho obrazku v px', required=True)
args_parser.add_argument('-ss','--super_size', type = int, help = 'Sirka a vyska obrazku pro supersampling px (musi byt vetsi nez velikost vysledneho obrazku)', default=256, required=False)
args_parser.add_argument('-ms','--min_size', type = int, help = 'Minimalni velikost objektu (musi byt mensi, nebo stejna, jako velikost obrazku)', default=50, required=False)
args_parser.add_argument('-to','--train_out', type = str, help = 'Nazev slozky, do ktere budou ulozeny obrazky train',default='train', required=False)
args_parser.add_argument('-t','--train_size', type = int, help = 'Pocet trenovacich dat', required=False)
args_parser.add_argument('-teo','--test_out', type = str, help = 'Nazev slozky, do ktere budou ulozeny obrazky test',default='test', required=False)
args_parser.add_argument('-te','--test_size', type = int, help = 'Pocet testovacich dat', required=False)
args_parser.add_argument('-c','--colour', help = 'Prida udaj o barve', action="store_true", required=False)
args_parser.add_argument('-p','--position', help = 'Prida udaj o pozici', action="store_true", required=False)
args_parser.add_argument('-z','--zoom', help = 'Prida udaj o velikosti', action="store_true", required=False)
args_parser.add_argument('-o','--rotation', help = 'Prida udaj o rotaci', action="store_true", required=False)
args_parser.add_argument('-a','--transformation', help = 'Prida udaj o transformaci', action="store_true", required=False)
args_parser.add_argument('-e','--everything', help = 'Prida vsechny udaje', action="store_true", required=False)
args_parser.add_argument('-rs','--random-size', help = 'Nahodna velikost, jinak velikost min_size', action="store_true", required=False)


try:
    args = vars(args_parser.parse_args())
except:
    sys.exit(1)


if (args['size'] <= 10):
    print("Prilis mala velikost obrazku(musi byt, vetsi nez 10). (--help)")
    sys.exit(1)

if (args['size'] > args['super_size']):
    print("Super_size musi byt vetsi, nebo rovno size (--help)")
    sys.exit(1)

if (args['min_size'] <= args['size']):
    print("Minimalni velikost, musi byt mensi, nebo rovna velikosti obrazku")
    sys.exit(1)


if (args['everything']):
    args['colour'] = True
    args['position'] = True
    args['zoom'] = True
    args['rotation'] = True
    args['transformation'] = True


if args['train_size']:
    create_set(0)

if args['test_size']:
    create_set(1)

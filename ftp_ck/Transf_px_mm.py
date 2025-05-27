def transf_px_mm(dims,w_mm,h_mm,desfase_cam):

  centers = []
  new_dims = []
  print(dims)
  for i in dims:
    centers.append((i[2],i[3]))
    new_dims.append((i[0],i[1]))

  escala = []
  for n in new_dims:
      #print(n[0])
      escala.append((w_mm/n[0],h_mm/n[1]))
  centers_mm = []
  for i in range(len(centers)):
      x_mm = (centers[i][0]-(2560/2))*(escala[i][0]+escala[i][1])/2
      y_mm = ((centers[i][1]-(2048/2))*((escala[i][0]+escala[i][1])/2))-desfase_cam

      centers_mm.append((x_mm,y_mm))
  return centers_mm,escala

def diferencia_angles(dims):
    angles=[]
    a = 0
    for i in dims:
      angles.append(i[4])

    print("angles ===== ",angles)

    dif_angles = []
    for a in angles:
      if a<200 and a>160:
        print("a ", a)
        dif_angles.append((90,a-180))
      elif a<110 and a>70:
        print("a ", a)
        dif_angles.append((0,a-90))
      elif a>-20 and a<20:
        dif_angles.append((90,a))
      a = a + 1
    return dif_angles


def transform_pallet_center(dims, desfase_cam, w_mm, h_mm):
  center = []
  escalax = dims['w']/w_mm
  escalay = dims['h']/h_mm
  prom_escala = (escalax+escalay)/2
  cx = (dims['cx'] - 2560/2)/prom_escala
  cy = ((dims['cy'] - 2048/2)/prom_escala )
  center.append(cx)
  center.append(cy)
  return center
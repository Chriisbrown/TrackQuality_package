import bitstring as bs
import random
import pandas
import numpy as np


TrackWord_config = {'InvR':      {'nbins':2**15,'min':-0.00855,'max':0.00855,"Signed":True ,'split':(15,0)},
                    'Phi':       {'nbins':2**12,'min':-1.026,  'max':1.026,  "Signed":False,'split':(11,0)},
                    'TanL':      {'nbins':2**16,'min':-7,      'max':7,      "Signed":True, 'split':(16,3)},
                    'Z0':        {'nbins':2**12,'min':-31,     'max':31,     "Signed":True, 'split':(12,5)},
                    'D0':        {'nbins':2**13,'min':-15.4,   'max':15.4,   "Signed":True, 'split':(13,5)},
                    'Chi2rphi':  {'bins':[0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,np.inf]},
                    'Chi2rz':    {'bins':[0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,np.inf]},
                    'Bendchi2':  {'bins':[0,0.5,1.25,2,3,5,10,50,np.inf]},
                    'Hitpattern':{'nbins':2**7 ,'min':0,       'max':2**7,   "Signed":False,'split':(7 ,0)},
                    'MVA1':      {'nbins':2**3 ,'min':0,       'max':1,      "Signed":False,'split':(3 ,0)},
                    'OtherMVA':  {'nbins':2**6 ,'min':0,       'max':1,      "Signed":False,'split':(6 ,0)},
                    'Pt':        {'nbins':2**16,'min':0,       'max':2**10,  "Signed":False,'split':(16,0)},
                    'HWUz0':     {'nbins':2**8 ,'min':-15,     'max':15,     "Signed":False,'split':(8, 0)},
                    'HWUtanl':   {'nbins':2**16,'min':-7,      'max':7,      "Signed":False,'split':(16,0)},
                    'HWUeta' :   {'nbins':2**16,'min':-2.644120761058629, 'max':2.644120761058629,"Signed":False,'split':(16,0)}
                    }

def astype(bitarray, t):
  assert isinstance(bitarray, bs.BitArray), "bitarray must be a bitstring.BitArray"
  return getattr(bitarray, t)

class PatternFileDataObj:
  fields = ['Example', 'datavalid']
  lengths = [16, 1]
  types = ['int:16', 'uint:1']
  datalength = 64

  def __init__(self, data=None, hexstring=None):
    if not data is None and not hexstring is None: 
      raise("Cannot initialise with field data and hexstring simultaneously")
    if not data is None:
      self._data = pandas.DataFrame(data, pandas.Index([0]))
    if not hexstring is None:
      hexdata = bs.BitArray(hexstring)
      d = {}
      for i in range(len(self.fields)):
        l = self.lengths[i]
        ll = sum(self.lengths[:i]) # field offset into 64b hex
        t = self.types[i].split(':')[0]
        d[self.fields[i]] = astype(hexdata[self.datalength-ll-l:self.datalength-ll], t)
      self._data = pandas.DataFrame(d, pandas.Index([0]))

  def pack(self):
    fmt_str = ''
    # pad to self.datalength bits
    if sum(self.lengths) < self.datalength:
      fmt_str += 'uint:' + str(self.datalength - sum(self.lengths)) + ', '
    for fmt in self.types[::-1][1:]:
      fmt_str += fmt + ', '  
    data = np.array(self._data[self.fields].iloc[0]).tolist()[::-1][1:] # reverse the fields
    # the pad field
    if sum(self.lengths) < self.datalength:
      data = [0] + data
    return bs.pack(fmt_str, *data)

  def toVHex(self,initial_char='v'):
    return str(np.array(self._data['framevalid'])[0]) + initial_char + self.pack().hex

  def data(self):
    return self._data

class LinkWord1(PatternFileDataObj):
  datalength = 64
  fields = ["InvR1","Phi1","TanLInt1","TanLFrac1","Z0Int1","Z0Frac1","QMVA1","OtherMVA1",'framevalid']
  lengths = [15, 12, 4,12, 5,7,3,6,1]
  types = ['int:15','int:12','int:4','uint:12','uint:5','int:7',
           'uint:3','uint:6','uint:1' ]

class LinkWord2(PatternFileDataObj):
  datalength = 64
  fields = ["D0Int1","D0Frac1","Chi2rphi1","Chi2rz1","Bendchi21","Hitpattern1",'datavalid1',
            "D0Int2","D0Frac2","Chi2rphi2","Chi2rz2","Bendchi22","Hitpattern2",'datavalid2','framevalid']
  lengths = [6,7,4,4,3,7,1,7,6,4,4,3,7,1,1]
  types = ['int:6','uint:7','uint:4','uint:4','uint:3','uint:7','uint:1',
           'int:6','uint:7','uint:4','uint:4','uint:3','uint:7','uint:1','uint:1' ]

class LinkWord3(PatternFileDataObj):
  datalength = 64
  fields = ["InvR2","Phi2","TanLInt2","TanLFrac2","Z0Int2","Z0Frac2","QMVA2","OtherMVA2",'framevalid']
  lengths = [15, 12, 4,12, 5,7,3,6,1]
  types = ['int:15','int:12','int:4','uint:12','uint:5','int:7',
           'uint:3','uint:6','uint:1' ]

class TrackWord(PatternFileDataObj):
  datalength = 96
  fields = ["InvR","Phi","TanL","Z0","D0","Chi2rphi","Chi2rz","BendChi","HitPattern","QMVA","OtherMVA",'Datavalid',"framevalid"]
  lengths = [15, 12, 16,12,13,4,4,3,7,3,6,1,1]
  types = ['int:15','int:12','int:16','int:12','int:13','uint:4',
           'uint:4','uint:3','uint:7','uint:3','uint:6','uint:1','uint:1']

def random_trks():
  ''' Make a track with random variables '''
  InvR1 = random.randint(-2**15,2**15-1)
  Phi1 = random.randint(-2**12, 2**12-1)
  TanL1 = random.randint(-2**16, 2**16-1)
  Z01 = random.randint(-2**12, 2**12-1)

  D01 = random.randint(-2**13, 2**13-1)
  Chi2rphi1 = random.randint(0, 2**4)
  Chi2rz1 = random.randint(0, 2**4)
  Bendchi21 = random.randint(0, 2**3)
  Hitpattern1 = random.randint(0, 2**7)
  QMVA1 = random.randint(0, 2**3)
  OtherMVA1= random.randint(0, 2**6)

  InvR2 = random.randint(-2**15,2**15-1)
  Phi2 = random.randint(-2**12, 2**12-1)
  TanL2 = random.randint(-2**16, 2**16-1)
  Z02 = random.randint(-2**12, 2**12-1)

  D02 = random.randint(-2**13, 2**13-1)
  Chi2rphi2 = random.randint(0, 2**4)
  Chi2rz2 = random.randint(0, 2**4)
  Bendchi22 = random.randint(0, 2**3)
  Hitpattern2 = random.randint(0, 2**7)
  QMVA2 = random.randint(0, 2**3)
  OtherMVA2 = random.randint(0, 2**6)
  
  Trk1 = LinkWord1({'InvR1':InvR1, 'Phi1':Phi1, 'TanL1':TanL1, 
                     'Z01':Z01,'QMVA1':QMVA1,'OtherMVA1':OtherMVA1, 'framevalid':1})

  Trk2 = LinkWord2({'D01':D01,'Chi2rphi1':Chi2rphi1,'Chi2rz1':Chi2rz1,'Bendchi21':Bendchi21,'Hitpattern1':Hitpattern1,'datavalid1':1, 
                    'D02':D02,'Chi2rphi2':Chi2rphi2,'Chi2rz2':Chi2rz2,'Bendchi22':Bendchi22,'Hitpattern2':Hitpattern2,'datavalid2':1,'framevalid':1})

  Trk3 = LinkWord3({'InvR2':InvR2, 'Phi2':Phi2, 'TanL2':TanL2, 
                     'Z02':Z02,'QMVA2':QMVA2,'OtherMVA2':OtherMVA2 ,'framevalid':1})
  

  return [Trk1,Trk2,Trk3]

def header(nlinks):
  txt = 'Board VX\n'
  txt += ' Quad/Chan :'
  for i in range(nlinks):
    quadstr = '        q{:02d}c{}      '.format(int(i/4), int(i%4))
    txt += quadstr
  txt += '\n      Link :'
  for i in range(nlinks):
    txt += '         {:03d}       '.format(i)
  txt += '\n'
  return txt

def frame(vhexdata, iframe, nlinks):
  assert(len(vhexdata) == nlinks), "Data length doesn't match expected number of links"
  txt = 'Frame {:04d} :'.format(iframe)
  for d in vhexdata:
    txt += ' ' + d
  txt += '\n'
  return txt

def empty_frames(n, istart, nlinks):
  ''' Make n empty frames for nlinks with starting frame number istart '''
  empty_data = '0v0000000000000000'
  empty_frame = [empty_data] * nlinks
  iframe = istart
  frames = []
  for i in range(n):
    frames.append(frame(empty_frame, iframe, nlinks))
    iframe += 1
  return frames

def toHWU(name,t):
  if (name == "Bendchi2") | (name == "Chi2rphi") |  (name =="Chi2rz"):
    return np.digitize(t,TrackWord_config[name]["bins"],right=True)-1
  else:

    import math
    '''Convert a track coordinate to Hardware Units'''
    intpart = TrackWord_config[name]['split'][1]
    n_bits = TrackWord_config[name]['split'][0]

    if TrackWord_config[name]['Signed']:
      signbit = np.signbit(t)
      if signbit:
        sign = -1
      else:
        sign = 1
      

      n_bits -= 1

    intnbins = 2**(intpart) -1
    fracnbins = 2**(n_bits-intpart) -1
    if intpart > 0:
      t = abs(t)
      fractional,integer = math.modf(t)
      
      tmin = 0
      tmax = TrackWord_config[name]['max']

      x = tmax if integer > tmax else integer
      x = tmin if integer < tmin else x
      # Get the bin index
      x = (x - tmin) / ((tmax - tmin) / intnbins)
      
      if name=="Z0":
        x = x
        y = np.copysign((fractional  * fracnbins) ,sign)
      else:
        x = np.copysign(x,sign)
        y = fractional  * fracnbins 

      return round(x),round(y)
      
    else:
      tmin = TrackWord_config[name]['min']
      tmax = TrackWord_config[name]['max']

      x = tmax if t > tmax else t
      x = tmin if t < tmin else x
      # Get the bin index
      x = (x - tmin) / ((tmax - tmin) / fracnbins)

      if TrackWord_config[name]['Signed']:
        t = abs(t)
        tmin = 0
        tmax = TrackWord_config[name]['max']

        x = tmax if t > tmax else t
        x = tmin if t < tmin else x
        # Get the bin index
        x = (x - tmin) / ((tmax - tmin) / fracnbins)
        x = np.copysign(x,sign)
        
      return round(x)
  
def HWUto(name,t):
  '''Convert a track coordinate in Hardware Units to '''
  tmin = TrackWord_config[name]['min']
  tmax = TrackWord_config[name]['max']

  x = t*(((tmax - tmin))/(TrackWord_config[name]['nbins'] - 1)) + tmin  
  return x

def eventDataFrameToPatternFile(event, nlinks=18, nframes=108, ninitialframes=17, doheader=True, startframe=0, weight='pt2'):
  '''Write a pattern file for an event dataframe.
  Tracks are assigned to links randomly
  '''
  # Push the tracks for each link into a list
  links = [] 
  for i in range(int(nlinks/2)):
    
    trks = event[event['phiSector'] == i]
    for j in [True,False]:
      LW1 = []
      splittrks = trks[np.signbit(trks['eta']) == j]
      ntrks = len(splittrks)
      for k in range(int((ntrks+1)/2)):
        if 2*k + 1 < ntrks:
            LW1.append(LinkWord1({'InvR1':toHWU("InvR",splittrks.iloc[k*2]["InvR"]), 
                                  #'InvR1':splittrks.iloc[k*2]["InvR"],
                                  'Phi1':toHWU("Phi",splittrks.iloc[k*2]["Sector_Phi"]),
                                  'TanLInt1':toHWU("TanL",splittrks.iloc[k*2]["TanL"])[0], 
                                  'TanLFrac1':toHWU("TanL",splittrks.iloc[k*2]["TanL"])[1], 
                                  'Z0Int1':toHWU("Z0",splittrks.iloc[k*2]["z0"])[0],
                                  'Z0Frac1':toHWU("Z0",splittrks.iloc[k*2]["z0"])[1],
                                  'QMVA1':toHWU("MVA1",splittrks.iloc[k*2]["MVA"]),
                                  'OtherMVA1':toHWU("OtherMVA",splittrks.iloc[k*2]["otherMVA"]),
                                  'framevalid':1}).toVHex())
            LW1.append(LinkWord2({'D0Int2':toHWU("D0",splittrks.iloc[k*2]["d0"])[0],
                                  'D0Frac2':toHWU("D0",splittrks.iloc[k*2]["d0"])[1],
                                  'Chi2rphi2':toHWU("Chi2rphi",splittrks.iloc[k*2]["chi2rphi"]),
                                  'Chi2rz2':toHWU("Chi2rz",splittrks.iloc[k*2]["chi2rz"]),
                                  'Bendchi22':toHWU("Bendchi2",splittrks.iloc[k*2]["bendchi2"]),
                                  'Hitpattern2':toHWU("Hitpattern",splittrks.iloc[k*2]["hitpattern"]),
                                  'datavalid2':1, 
                                  'D0Int1':toHWU("D0",splittrks.iloc[k*2+1]["d0"])[0],
                                  'D0Frac1':toHWU("D0",splittrks.iloc[k*2+1]["d0"])[1],
                                  'Chi2rphi1':toHWU("Chi2rphi",splittrks.iloc[k*2+1]["chi2rphi"]),
                                  'Chi2rz1':toHWU("Chi2rz",splittrks.iloc[k*2+1]["chi2rz"]),
                                  'Bendchi21':toHWU("Bendchi2",splittrks.iloc[k*2+1]["bendchi2"]),
                                  'Hitpattern1':toHWU("Hitpattern",splittrks.iloc[k*2+1]["hitpattern"]),
                                  'datavalid1':1,
                                  'framevalid':1}).toVHex())
            LW1.append(LinkWord3({'InvR2':toHWU("InvR",splittrks.iloc[k*2+1]["InvR"]),
                                  #'InvR2':splittrks.iloc[k*2+1]["InvR"],
                                  'Phi2':toHWU("Phi",splittrks.iloc[k*2+1]["Sector_Phi"]), 
                                  'TanLInt2':toHWU("TanL",splittrks.iloc[k*2+1]["TanL"])[0], 
                                  'TanLFrac2':toHWU("TanL",splittrks.iloc[k*2+1]["TanL"])[1], 
                                  'Z0Int2':toHWU("Z0",splittrks.iloc[k*2+1]["z0"])[0],
                                  'Z0Frac2':toHWU("Z0",splittrks.iloc[k*2+1]["z0"])[1],
                                  'QMVA2':toHWU("MVA1",splittrks.iloc[k*2+1]["MVA"]),
                                  'OtherMVA2':toHWU("OtherMVA",splittrks.iloc[k*2+1]["otherMVA"]),
                                  'framevalid':1}).toVHex())
        else:
          LW1.append(LinkWord1({'InvR1':toHWU("InvR",splittrks.iloc[2*k]["InvR"]), 
                              #'InvR1':splittrks.iloc[2*k]["InvR"], 
                              'Phi1':toHWU("Phi",splittrks.iloc[2*k]["Sector_Phi"]),
                              'TanLInt1':toHWU("TanL",splittrks.iloc[2*k]["TanL"])[0], 
                              'TanLFrac1':toHWU("TanL",splittrks.iloc[2*k]["TanL"])[1], 
                              'Z0Int1':toHWU("Z0",splittrks.iloc[2*k]["z0"])[0],
                              'Z0Frac1':toHWU("Z0",splittrks.iloc[2*k]["z0"])[1],
                              'QMVA1':toHWU("MVA1",splittrks.iloc[2*k]["MVA"]),
                              'OtherMVA1':toHWU("OtherMVA",splittrks.iloc[2*k]["otherMVA"]),
                              'framevalid':1}).toVHex())

          LW1.append(LinkWord2({'D0Int2':toHWU("D0",splittrks.iloc[2*k]["d0"])[0],
                                'D0Frac2':toHWU("D0",splittrks.iloc[2*k]["d0"])[1],
                                'Chi2rphi2':toHWU("Chi2rphi",splittrks.iloc[2*k]["chi2rphi"]),
                                'Chi2rz2':toHWU("Chi2rz",splittrks.iloc[2*k]["chi2rz"]),
                                'Bendchi22':toHWU("Bendchi2",splittrks.iloc[2*k]["bendchi2"]),
                                'Hitpattern2':toHWU("Hitpattern",splittrks.iloc[2*k]["hitpattern"]),
                                'datavalid2':1,
                                'D0Int1':0,'D0Frac1':0,'Chi2rphi1':0,'Chi2rz1':0,
                                'Bendchi21':0,'Hitpattern1':0,
                                'datavalid1':0,'framevalid':1}).toVHex())

          LW1.append(LinkWord3({'InvR2':0, 'Phi2':0, 'TanLInt2':0,'TanLFrac2':0, 'Z0Int2':0,'Z0Frac2':0,'QMVA2':0,'OtherMVA2':0,'framevalid':1}).toVHex())

    # Pad up to the frame length
      for m in range(int((nframes - len(LW1))/3)):
        LW1.append(LinkWord1({'InvR1':0, 'Phi1':0, 'TanLInt1':0,  'TanLFrac1':0, 
                            'Z0Int1':-0,'Z0Frac1':0,'QMVA1':0,'OtherMVA1':0,'framevalid':1}).toVHex())

        LW1.append(LinkWord2({'D0Int1':0,'D0Frac1':0,'Chi2rphi1':0,'Chi2rz1':0,
                            'Bendchi21':0,'Hitpattern1':0,
                            'datavalid1':0, 'D0Int2':0,'D0Frac2':0,'Chi2rphi2':0,'Chi2rz2':0,
                            'Bendchi22':0,'Hitpattern2':0,
                            'datavalid2':0,'framevalid':1}).toVHex())

        LW1.append(LinkWord3({'InvR2':0, 'Phi2':0, 'TanLInt2':0, 'TanLFrac2':0, 
                            'Z0Int2':-0,'Z0Frac2':0,'QMVA2':0,'OtherMVA2':0,'framevalid':1}).toVHex())


      links.append(LW1)

  # Put empty frames on the remaining links
  empty_data = '1v0000000000000000'
  empty_link = [empty_data] * nframes
  for i in range(72 - nlinks):
    links.append(empty_link)


  links = np.array(links)
  frames = links.transpose()
  frames = [frame(f, i+startframe+ninitialframes, 72) for i, f in enumerate(frames)] # +8 because there will be 8 frames of header

  ret = []
  if doheader:
    ret = [header(72)]
  
  return ret + empty_frames(ninitialframes, startframe, 72) + frames# + empty_frames(16, 8 + nframes, 72)

def writepfile(filename, events, ninitialframes, weight='pt2'):
  doheader = True
  startframe = 0
  with open(filename, 'w') as pfile:
    for i,event in enumerate(events):
      if i > 0:
        doheader = False
        startframe += 108 + ninitialframes # The data and inter-event gap
      for frame in eventDataFrameToPatternFile(event, doheader=doheader, startframe=startframe,ninitialframes=ninitialframes,weight=weight):
        pfile.write(frame)
    pfile.close()

def writemultipfile(filename, events, ninitialframes,weight='pt2'):
  doheader = True
  startframe = 0
  batches = int(len(events)/7)
  for batch in range(batches):
    events_batch = events[7*batch:7*batch+7].copy()
    startframe = 0
    doheader = True

    with open(filename+str(batch)+".txt", 'w') as pfile:
      for i,event in enumerate(events_batch):

        if i > 0:
          doheader = False
          startframe += 108 + ninitialframes # The data and inter-event gap
        for frame in eventDataFrameToPatternFile(event, doheader=doheader, startframe=startframe,ninitialframes=ninitialframes, weight=weight):
          pfile.write(frame)
      pfile.close()


/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.tsfile.encoding.encoder;

import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.exception.encoding.TsFileEncodingException;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.utils.BitMap;
import org.apache.tsfile.utils.ReadWriteForEncodingUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Encoder for float or double value using rle or two-diff according to following grammar.
 *
 * <pre>{@code
 * float encoder: <maxPointvalue> <encoded-data>
 * maxPointvalue := number for accuracy of decimal places, store as unsigned var int
 * encoded-data := same as encoder's pattern
 * }</pre>
 */
public class FloatEncoder extends Encoder {

  private Encoder encoder;

  /** number for accuracy of decimal places. */
  private int maxPointNumber;

  /** maxPointValue = 10^(maxPointNumber). */
  private double maxPointValue;

  /** flag to check whether maxPointNumber is saved in the stream. */
  private boolean isMaxPointNumberSaved;

  private final List<Boolean> useMaxPointNumber;

  public FloatEncoder(TSEncoding encodingType, TSDataType dataType, int maxPointNumber) {
    super(encodingType);
    this.maxPointNumber = maxPointNumber;
    calculateMaxPointNum();
    isMaxPointNumberSaved = false;
    useMaxPointNumber = new ArrayList<>();
    if (encodingType == TSEncoding.RLE) {
      if (dataType == TSDataType.FLOAT) {
        encoder = new IntRleEncoder();
      } else if (dataType == TSDataType.DOUBLE) {
        encoder = new LongRleEncoder();
      } else {
        throw new TsFileEncodingException(
            String.format("data type %s is not supported by FloatEncoder", dataType));
      }
    } else if (encodingType == TSEncoding.TS_2DIFF) {
      if (dataType == TSDataType.FLOAT) {
        encoder = new DeltaBinaryEncoder.IntDeltaEncoder();
      } else if (dataType == TSDataType.DOUBLE) {
        encoder = new DeltaBinaryEncoder.LongDeltaEncoder();
      } else {
        throw new TsFileEncodingException(
            String.format("data type %s is not supported by FloatEncoder", dataType));
      }
    } else if (encodingType == TSEncoding.RLBE) {
      if (dataType == TSDataType.FLOAT) {
        encoder = new IntRLBE();
      } else if (dataType == TSDataType.DOUBLE) {
        encoder = new LongRLBE();
      } else {
        throw new TsFileEncodingException(
            String.format("data type %s is not supported by FloatEncoder", dataType));
      }
    } else {
      throw new TsFileEncodingException(
          String.format("%s encoding is not supported by FloatEncoder", encodingType));
    }
  }

  @Override
  public void encode(float value, ByteArrayOutputStream out) {
    saveMaxPointNumber(out);
    int valueInt = convertFloatToInt(value);
    encoder.encode(valueInt, out);
  }

  @Override
  public void encode(double value, ByteArrayOutputStream out) {
    saveMaxPointNumber(out);
    long valueLong = convertDoubleToLong(value);
    encoder.encode(valueLong, out);
  }

  private void calculateMaxPointNum() {
    if (maxPointNumber <= 0) {
      maxPointNumber = 0;
      maxPointValue = 1;
    } else {
      maxPointValue = Math.pow(10, maxPointNumber);
    }
  }

  private int convertFloatToInt(float value) {
    if (value * maxPointValue > Integer.MAX_VALUE || value * maxPointValue < Integer.MIN_VALUE) {
      useMaxPointNumber.add(false);
      return Math.round(value);
    } else {
      useMaxPointNumber.add(true);
      return (int) Math.round(value * maxPointValue);
    }
  }

  private long convertDoubleToLong(double value) {
    if (value * maxPointValue > Long.MAX_VALUE || value * maxPointValue < Long.MIN_VALUE) {
      useMaxPointNumber.add(false);
      return Math.round(value);
    } else {
      useMaxPointNumber.add(true);
      return Math.round(value * maxPointValue);
    }
  }

  @Override
  public void flush(ByteArrayOutputStream out) throws IOException {
    encoder.flush(out);
    if (pointsNotUseMaxPointNumber()) {
      byte[] ba = out.toByteArray();
      out.reset();
      ReadWriteForEncodingUtils.writeUnsignedVarInt(Integer.MAX_VALUE, out);
      BitMap bitMap = new BitMap(useMaxPointNumber.size());
      for (int i = 0; i < useMaxPointNumber.size(); i++) {
        if (useMaxPointNumber.get(i)) {
          bitMap.mark(i);
        }
      }
      ReadWriteForEncodingUtils.writeUnsignedVarInt(useMaxPointNumber.size(), out);
      out.write(bitMap.getByteArray());
      out.write(ba);
    }
    reset();
  }

  private void reset() {
    isMaxPointNumberSaved = false;
    useMaxPointNumber.clear();
  }

  private boolean pointsNotUseMaxPointNumber() {
    for (boolean info : useMaxPointNumber) {
      if (!info) {
        return true;
      }
    }
    return false;
  }

  private void saveMaxPointNumber(ByteArrayOutputStream out) {
    if (!isMaxPointNumberSaved) {
      ReadWriteForEncodingUtils.writeUnsignedVarInt(maxPointNumber, out);
      isMaxPointNumberSaved = true;
    }
  }

  @Override
  public int getOneItemMaxSize() {
    return encoder.getOneItemMaxSize();
  }

  @Override
  public long getMaxByteSize() {
    return encoder.getMaxByteSize();
  }
}
